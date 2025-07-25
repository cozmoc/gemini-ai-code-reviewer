import json
import os
import logging
import time
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
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 5))  # number of hunks per batch
RETRY_LIMIT = int(os.environ.get('RETRY_LIMIT', 3))
BACKOFF_BASE = float(os.environ.get('BACKOFF_BASE', 1.0))  # seconds

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

    # Flatten hunks by file
    for pf in parsed:
        if not pf.path or pf.path == "/dev/null":
            continue
        hunks = list(pf)
        # Batch hunks
        for i in range(0, len(hunks), BATCH_SIZE):
            batch = hunks[i : i + BATCH_SIZE]
            prompt = create_batch_prompt(pf.path, batch, pr_details)
            ai_reviews = get_ai_response(prompt)
            # Map reviews back to hunks
            for r in ai_reviews:
                try:
                    file_index = int(r.get('hunkIndex', 0))
                    rel = int(r['lineNumber'])
                    hunk = batch[file_index]
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

    # Sort and filter
    reviews.sort(key=lambda x: x['severity'], reverse=True)
    suggestions = [r for r in reviews if r['type'] == 'suggestion'][:MAX_SUGGESTIONS]
    comments = [r for r in reviews if r['type'] == 'comment'][:MAX_COMMENTS]
    chosen = suggestions + comments
    chosen.sort(key=lambda x: x['severity'], reverse=True)

    logger.info(f"Returning {len(chosen)} review comments")
    return chosen


def create_batch_prompt(path: str, hunks: List[Any], pr: PRDetails) -> str:
    diff_blocks = []
    for idx, hunk in enumerate(hunks):
        content = '\n'.join(hunk.source)
        diff_blocks.append(f"--- Hunk {idx} ---\n```diff\n{content}\n```")
    blocks = '\n'.join(diff_blocks)
    return f"""
You are a senior code reviewer.
Review the PR changes in `{path}`, batch of hunks, and identify **critical or subtle issues**.
Prepend each review with the corresponding hunk index in field `hunkIndex`.
Return JSON with:
{{
  "reviews": [
    {{
      "hunkIndex": <index>,
      "lineNumber": <relative_line>,
      "reviewComment": "<insightful comment>",
      "severity": <1-5>,
      "type": "suggestion" | "comment"
    }}
  ]
}}

PR Title: {pr.title}
PR Description:
---
{pr.description or 'None'}
---
{blocks}"""


def get_ai_response(prompt: str) -> List[Dict[str, Any]]:
    model_name = os.environ.get('GEMINI_MODEL', 'gemini-1.0-small')
    model = Client.GenerativeModel(model_name)
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            logger.info(f"Sending prompt to Gemini API (attempt {attempt})")
            text = model.generate_content(
                prompt,
                generation_config={"max_output_tokens": 4096}
            ).text
            clean = text.strip().lstrip('```json').rstrip('```').strip()
            response = json.loads(clean)
            return response.get('reviews', [])
        except Exception as e:
            logger.error(f"Error from Gemini API on attempt {attempt}: {e}")
            if attempt < RETRY_LIMIT:
                delay = BACKOFF_BASE * (2 ** (attempt - 1))
                logger.info(f"Retrying after {delay:.1f}s")
                time.sleep(delay)
            else:
                logger.error("Max retries reached, returning empty reviews")
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
        return

    ev = json.load(open(os.environ['GITHUB_EVENT_PATH']))
    if not ev.get('issue', {}).get('pull_request'):
        return

    repo = gh.get_repo(f"{pr.owner}/{pr.repo}")
    pull = repo.get_pull(pr.pull_number)
    gh_files = pull.get_files()

    reviews = []
    exclude_patterns = [p.strip() for p in os.environ.get('INPUT_EXCLUDE', '').split(',') if p.strip()]

    for gh_file in gh_files:
        if not gh_file.patch:
            continue

        header = (
            f"diff --git a/{gh_file.filename} b/{gh_file.filename}\n"
            f"--- a/{gh_file.filename}\n"
            f"+++ b/{gh_file.filename}\n"
        )
        try:
            parsed = PatchSet(header + gh_file.patch)
        except Exception:
            continue

        parsed = [pf for pf in parsed if pf.path == gh_file.filename]
        parsed = [pf for pf in parsed if not any(fnmatch.fnmatch(pf.path, pat) for pat in exclude_patterns)]

        reviews.extend(analyze_code(parsed, pr))

    if reviews:
        create_review_comment(pr.owner, pr.repo, pr.pull_number, reviews)


if __name__ == '__main__':
    main()
