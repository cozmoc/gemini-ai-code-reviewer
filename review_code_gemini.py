import json
import os
import logging
import time
from typing import List, Dict, Any, Tuple
import google.generativeai as Client
from github import Github
import fnmatch
from unidiff import PatchSet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash-lite-001')
MAX_SUGGESTIONS = 1
MAX_COMMENTS = 4
HUNK_BATCH_SIZE = int(os.environ.get('INPUT_BATCH_SIZE', 5))
MAX_RETRIES = int(os.environ.get('INPUT_MAX_RETRIES', 3))
BACKOFF_FACTOR = float(os.environ.get('INPUT_BACKOFF_FACTOR', 1.0))

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


def get_ai_response(prompt: str) -> List[Dict[str, Any]]:
    """
    Send prompt to Gemini API with retries and exponential backoff.
    """
    model = Client.GenerativeModel(GEMINI_MODEL)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Sending prompt to Gemini API (attempt {attempt})")
            result = model.generate_content(
                prompt,
                generation_config={"max_output_tokens": 8192}
            )
            text = result.text
            clean = text.strip().lstrip('```json').rstrip('```').strip()
            response = json.loads(clean)

            if isinstance(response, list):
                return response
            elif isinstance(response, dict):
                return response.get('reviews', [])
            else:
                logger.warning("Unexpected response format from Gemini API")
                return []
        except Exception as e:
            logger.warning(f"Error from Gemini API (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                sleep_time = BACKOFF_FACTOR * (2 ** (attempt - 1))
                logger.info(f"Retrying after {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                logger.error("Max retries reached. Giving up.")
                return []

def analyze_code(parsed: List[Any], pr_details: PRDetails) -> List[Dict[str, Any]]:
    """
    Analyze code changes in batches of hunks to reduce API calls.
    """
    logger.info(f"Analyzing code for PR: {pr_details.pull_number} in batches of {HUNK_BATCH_SIZE}")
    reviews: List[Dict[str, Any]] = []

    # Collect all hunks with associated file paths
    hunks: List[Tuple[str, Any]] = []
    for pf in parsed:
        if not pf.path or pf.path == "/dev/null":
            logger.info(f"Skipping deleted or invalid path: {pf.path}")
            continue
        for hunk in pf:
            hunks.append((pf.path, hunk))

    # Process in batches
    for i in range(0, len(hunks), HUNK_BATCH_SIZE):
        batch = hunks[i:i + HUNK_BATCH_SIZE]
        batch_prompt = create_batch_prompt(batch, pr_details)
        ai_reviews = get_ai_response(batch_prompt)
        logger.debug(f"Received {len(ai_reviews)} review(s) for batch {i // HUNK_BATCH_SIZE + 1}")

        # Match reviews back to each hunk context
        for r in ai_reviews:
            try:
                path = r['path']
                rel = int(r['lineNumber'])
                # find the corresponding hunk
                for pf_path, hunk in batch:
                    if pf_path == path:
                        file_line = hunk.target_start + rel - 1
                        reviews.append({
                            'path': path,
                            'side': 'RIGHT',
                            'line': file_line,
                            'body': wrap_body(r),
                            'severity': int(r.get('severity', 0)),
                            'type': r.get('type', 'comment')
                        })
                        break
            except Exception as e:
                logger.warning(f"Invalid review object: {r} - {e}")

    # Select top suggestions and comments
    reviews.sort(key=lambda x: x['severity'], reverse=True)
    suggestions = [r for r in reviews if r['type'] == 'suggestion'][:MAX_SUGGESTIONS]
    comments = [r for r in reviews if r['type'] == 'comment'][:MAX_COMMENTS]
    chosen = suggestions + comments
    chosen.sort(key=lambda x: x['severity'], reverse=True)

    logger.info(f"Returning {len(chosen)} review comments")
    return chosen


def create_batch_prompt(batch: List[Tuple[str, Any]], pr: PRDetails) -> str:
    """
    Create a combined prompt for multiple hunks across one or more files.
    """
    diffs = []
    for path, hunk in batch:
        diff_header = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
        content = '\n'.join(hunk.source)
        diffs.append(diff_header + content)

    diff_block = '\n'.join(diffs)
    return f"""
You are a senior code reviewer.
Review the following PR hunks and identify **critical or subtle issues** in logic, edge cases, data handling, and correctness.

Return JSON objects with a 'path' field and 'reviews' array per hunk.

PR Title: {pr.title}
PR Description:
---
{pr.description or 'None'}
---
```diff
{diff_block}
```"""


def wrap_body(r: Dict[str, Any]) -> str:
    if r.get('type') == 'suggestion':
        return f"""```suggestion
{r['reviewComment']}
```"""
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
