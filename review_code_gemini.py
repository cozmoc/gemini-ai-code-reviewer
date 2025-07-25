import json
import os
import logging
from typing import List, Dict, Any
import google.generativeai as Client
from github import Github
import requests
import fnmatch
from unidiff import PatchSet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
MAX_SUGGESTIONS = 1
MAX_COMMENTS = 4

# Initialize clients
gh = Github(GITHUB_TOKEN)
Client.configure(api_key=os.environ.get('GEMINI_API_KEY'))

class PRDetails:
    def __init__(self, owner: str, repo: str, pull_number: int, title: str, description: str):
        self.owner = owner
        self.repo = repo
        self.pull_number = pull_number
        self.title = title
        self.description = description


def get_pr_details() -> PRDetails:
    logger.info("Fetching PR details from GITHUB_EVENT_PATH")
    with open(os.environ["GITHUB_EVENT_PATH"], "r") as f:
        data = json.load(f)

    if "issue" in data and data["issue"].get("pull_request"):
        num = data["issue"]["number"]
    else:
        num = data["number"]

    full = data["repository"]["full_name"]
    owner, repo = full.split("/")
    pr = gh.get_repo(full).get_pull(num)

    logger.info(f"Loaded PR #{num} from {owner}/{repo}")
    return PRDetails(owner, repo, num, pr.title, pr.body)


def analyze_code(parsed: PatchSet, pr_details: PRDetails) -> List[Dict[str, Any]]:
    logger.info(f"Analyzing code for PR: {pr_details.pull_number}")
    reviews = []

    for pf in parsed:
        if not pf.path or pf.path == "/dev/null":
            logger.info(f"Skipping deleted or invalid path: {pf.path}")
            continue

        for hunk in pf:
            prompt = create_prompt(pf.path, hunk, pr_details)
            logger.debug(f"Sending prompt for {pf.path}")
            ai_reviews = get_ai_response(prompt)
            logger.debug(f"Received {len(ai_reviews)} review(s) from AI")

            for r in ai_reviews:
                try:
                    rel = int(r['lineNumber'])
                    file_line = hunk.target_start + rel - 1
                    reviews.append({
                        'path': pf.path,
                        'side': 'RIGHT',
                        'line': file_line,
                        'body': wrap_body(r),
                        'severity': int(r.get('severity', 0)),
                        'type': r.get('type', 'comment')
                    })
                except Exception as e:
                    logger.warning(f"Invalid review object: {r} - {e}")

    reviews.sort(key=lambda x: x['severity'], reverse=True)
    suggestions = [r for r in reviews if r['type'] == 'suggestion'][:MAX_SUGGESTIONS]
    comments = [r for r in reviews if r['type'] == 'comment'][:MAX_COMMENTS]
    chosen = suggestions + comments
    chosen.sort(key=lambda x: x['severity'], reverse=True)

    logger.info(f"Returning {len(chosen)} review comments")
    return chosen


def create_prompt(path: str, hunk: Any, pr: PRDetails) -> str:
    content = '\n'.join(hunk.source)
    return f"""
You are a senior code reviewer.
Review the PR changes in `{path}` and identify **critical or subtle issues** in logic, state, correctness, edge cases, and data handling.
Avoid shallow comments like "consider a try-catch" or stylistic nitpicks.

Return JSON in the format:
{{
  "reviews": [
    {{
      "lineNumber": <relative_line>,
      "reviewComment": "<insightful and actionable comment>",
      "severity": <1-5>,  // 1=minor, 5=critical
      "type": "suggestion" | "comment"
    }}
  ]
}}

Guidelines:
- Use `"type": "suggestion"` for code changes, `"comment"` for observations.
- For suggestions, wrap them in a ```suggestion``` code block.
- Focus on correctness, data integrity, race conditions, edge cases, and non-obvious bugs.
- Ignore irrelevant concerns like missing logging, try/catch, or formatting.
- Think like someone who would block a PR for a production-critical bug.

PR Title: {pr.title}
PR Description:
---
{pr.description or 'None'}
---
```diff
{content}
```"""



def get_ai_response(prompt: str) -> List[Dict[str, Any]]:
    model = Client.GenerativeModel(os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash-001'))
    try:
        logger.info("Sending prompt to Gemini API")
        text = model.generate_content(prompt, generation_config={"max_output_tokens": 8192}).text
        clean = text.strip().lstrip('```json').rstrip('```').strip()
        response = json.loads(clean)
        return response.get('reviews', [])
    except Exception as e:
        logger.error(f"Error from Gemini API: {e}")
        return []


def wrap_body(r: Dict[str, Any]) -> str:
    if r.get('type') == 'suggestion':
        return f"```suggestion\n{r['reviewComment']}\n```"
    return r['reviewComment']


def create_review_comment(owner: str, repo: str, num: int, reviews: List[Dict[str, Any]]):
    logger.info(f"Posting {len(reviews)} comments to PR #{num}")
    pr = gh.get_repo(f"{owner}/{repo}").get_pull(num)
    gh_comments = [
        {'path': rev['path'], 'line': rev['line'], 'side': rev['side'], 'body': rev['body']}
        for rev in reviews
    ]
    pr.create_review(body="AI Review", comments=gh_comments, event="COMMENT")
    logger.info("Review posted successfully")


def main():
    logger.info("AI PR Review bot started")
    pr = get_pr_details()

    if os.environ.get('GITHUB_EVENT_NAME') != 'issue_comment':
        logger.info("Not triggered by issue_comment event, exiting.")
        return

    ev = json.load(open(os.environ['GITHUB_EVENT_PATH']))
    if not ev.get('issue', {}).get('pull_request'):
        logger.info("No pull_request found in event data, exiting.")
        return

    repo = gh.get_repo(f"{pr.owner}/{pr.repo}")
    pull = repo.get_pull(pr.pull_number)
    gh_files = pull.get_files()

    reviews = []
    exclude_patterns = [p.strip() for p in os.environ.get('INPUT_EXCLUDE', '').split(',') if p.strip()]
    logger.info(f"Excluding files matching patterns: {exclude_patterns}")

    for gh_file in gh_files:
        if not gh_file.patch:
            logger.info(f"Skipping file without patch: {gh_file.filename}")
            continue

        header = (
            f"diff --git a/{gh_file.filename} b/{gh_file.filename}\n"
            f"--- a/{gh_file.filename}\n"
            f"+++ b/{gh_file.filename}\n"
        )
        try:
            parsed = PatchSet(header + gh_file.patch)
        except Exception as e:
            logger.warning(f"Failed to parse patch for {gh_file.filename}: {e}")
            continue

        parsed = [pf for pf in parsed if pf.path == gh_file.filename]
        parsed = [pf for pf in parsed if not any(fnmatch.fnmatch(pf.path, pat) for pat in exclude_patterns)]

        reviews.extend(analyze_code(parsed, pr))

    if reviews:
        create_review_comment(pr.owner, pr.repo, pr.pull_number, reviews)
    else:
        logger.info("No reviews to post")

if __name__ == '__main__':
    main()
