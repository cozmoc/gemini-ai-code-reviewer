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
GITHUB_TOKEN    = os.environ["GITHUB_TOKEN"]
GEMINI_MODEL    = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash-lite-001')
MAX_SUGGESTIONS = 1
MAX_COMMENTS    = 4
HUNK_BATCH_SIZE = int(os.environ.get('INPUT_BATCH_SIZE', 5))
MAX_RETRIES     = int(os.environ.get('INPUT_MAX_RETRIES', 3))
BACKOFF_FACTOR  = float(os.environ.get('INPUT_BACKOFF_FACTOR', 1.0))

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

    num = data["issue"]["number"] if data.get("issue", {}).get("pull_request") else data["number"]
    full = data["repository"]["full_name"]
    owner, repo = full.split("/")
    pr = gh.get_repo(full).get_pull(num)

    logger.info(f"Loaded PR #{num} from {owner}/{repo}")
    return PRDetails(owner, repo, num, pr.title, pr.body)

def get_ai_response(prompt: str) -> List[Dict[str, Any]]:
    model = Client.GenerativeModel(GEMINI_MODEL)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = model.generate_content(
                prompt,
                generation_config={"max_output_tokens": 8192}
            )
            text = result.text or ""
            if not text.strip():
                logger.warning(f"Empty response from Gemini API on attempt {attempt}")
                return []

            # Extract only the JSON content
            clean = text.strip()
            if clean.startswith("```json"):
                clean = clean[len("```json"):].strip()
            if clean.endswith("```"):
                clean = clean[:-3].strip()

            try:
                payload = json.loads(clean)
            except json.JSONDecodeError as je:
                logger.warning(
                    f"Could not parse JSON from Gemini API response (attempt {attempt}): {je}\n"
                    f"Raw response:\n{text!r}"
                )
                return []

            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict) and 'reviews' in payload:
                return payload['reviews']

            logger.warning(f"Unexpected payload format from Gemini API: {payload!r}")
            return []

        except Exception as e:
            logger.warning(f"Gemini API error (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                sleep_time = BACKOFF_FACTOR * (2 ** (attempt - 1))
                logger.info(f"Retrying after {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                return []


def analyze_code(parsed: List[Any], pr: PRDetails) -> List[Dict[str, Any]]:
    logger.info(f"Analyzing PR #{pr.pull_number} in batches of {HUNK_BATCH_SIZE}")
    SEVERITY_MAP = {"critical": 4, "major": 3, "minor": 2, "info": 1}

    # Flatten hunks
    hunks: List[Tuple[str, Any]] = []
    for pf in parsed:
        if not pf.path or pf.path == "/dev/null":
            continue
        for h in pf:
            hunks.append((pf.path, h))

    reviews: List[Dict[str, Any]] = []
    for i in range(0, len(hunks), HUNK_BATCH_SIZE):
        batch = hunks[i : i + HUNK_BATCH_SIZE]
        prompt = create_batch_prompt(batch, pr)
        ai_blocks = get_ai_response(prompt)
        if not isinstance(ai_blocks, list):
            continue

        for block in ai_blocks:
            path = block.get('path')
            if not path or 'reviews' not in block:
                continue

            for r in block['reviews']:
                comment_text = r.get('comment') or r.get('description') or r.get('message')
                if not isinstance(comment_text, str):
                    continue
                try:
                    rel_line = int(r.get('line'))
                except:
                    continue

                raw_sev = r.get('severity', 0)
                sev = (
                    SEVERITY_MAP.get(str(raw_sev).lower(), 0)
                    if isinstance(raw_sev, str)
                    else int(raw_sev)
                )

                # Find matching hunk to compute 'position'
                for pf_path, hunk in batch:
                    if pf_path != path:
                        continue

                    diff_lines = [ln for ln in hunk if ln.is_added or ln.is_context]
                    position = None
                    for idx, ln in enumerate(diff_lines):
                        if ln.target_line_no == rel_line:
                            position = idx
                            break
                    if position is None:
                        logger.warning(f"No position match for {path}@{rel_line}")
                        continue

                    reviews.append({
                        'path':     path,
                        'position': position,
                        'body':     wrap_body({
                                         'type': r.get('type','comment'),
                                         'reviewComment': comment_text
                                     }),
                        'severity': sev
                    })
                    break

    reviews.sort(key=lambda x: x['severity'], reverse=True)
    suggestions = [r for r in reviews if 'suggestion' in r['body']][:MAX_SUGGESTIONS]
    comments    = [r for r in reviews if 'suggestion' not in r['body']][:MAX_COMMENTS]
    return sorted(suggestions + comments, key=lambda x: x['severity'], reverse=True)

def create_batch_prompt(batch: List[Tuple[str, Any]], pr: PRDetails) -> str:
    diffs = []
    for path, hunk in batch:
        header = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
        diffs.append(header + "\n".join(hunk.source))
    
    return f"""
You are a senior code reviewer.

Review the following PR hunks and return a JSON array. Each element must be an object like:
{{
  "path": "<file path>",
  "reviews": [
    {{
      "line": <line number>,
      "comment": "<review comment>",
      "type": "comment" | "suggestion",
      "severity": "critical" | "major" | "minor" | "info"
    }}
  ]
}}

Respond with **only** valid JSON. Do not include explanations, markdown, or prose.

PR Title: {pr.title}
PR Description:
---
{pr.description or 'None'}
---
```diff
{''.join(diffs)}
```"""

def wrap_body(r: Dict[str, Any]) -> str:
    if r.get('type') == 'suggestion':
        return f"```suggestion\n{r['reviewComment']}\n```"
    return r['reviewComment']

def create_review_comment(owner: str, repo: str, num: int, reviews: List[Dict[str, Any]]):
    logger.info(f"Posting {len(reviews)} comments to PR #{num}")
    pr = gh.get_repo(f"{owner}/{repo}").get_pull(num)
    gh_comments = [
        {'path': rev['path'], 'position': rev['position'], 'body': rev['body']}
        for rev in reviews
    ]
    pr.create_review(body="AI Review", comments=gh_comments, event="COMMENT")
    logger.info("Review posted successfully")

def main():
    logger.info("AI PR Review bot started")
    if os.environ.get('GITHUB_EVENT_NAME') != 'issue_comment':
        return

    pr = get_pr_details()
    ev = json.load(open(os.environ['GITHUB_EVENT_PATH']))
    if not ev.get('issue', {}).get('pull_request'):
        return

    repo = gh.get_repo(f"{pr.owner}/{pr.repo}")
    pull = repo.get_pull(pr.pull_number)
    exclude = [p.strip() for p in os.environ.get('INPUT_EXCLUDE','').split(',') if p.strip()]

    parsed = []
    for f in pull.get_files():
        if not f.patch:
            continue
        try:
            ps = PatchSet(f"diff --git a/{f.filename} b/{f.filename}\n{f.patch}")
        except Exception:
            continue
        ps = [pf for pf in ps if pf.path == f.filename]
        ps = [pf for pf in ps if not any(fnmatch.fnmatch(pf.path, pat) for pat in exclude)]
        parsed.extend(ps)

    reviews = analyze_code(parsed, pr)
    if reviews:
        create_review_comment(pr.owner, pr.repo, pr.pull_number, reviews)
    else:
        logger.info("No reviews to post")

if __name__ == '__main__':
    main()
