import json
import os
from typing import List, Dict, Any, Any as TypingAny
import google.generativeai as Client
from github import Github
import fnmatch
from unidiff import PatchSet

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
    with open(os.environ["GITHUB_EVENT_PATH"], "r") as f:
        data = json.load(f)
    if "issue" in data and data["issue"].get("pull_request"):
        num = data["issue"]["number"]
    else:
        num = data["number"]
    full = data["repository"]["full_name"]
    owner, repo = full.split("/")
    pr = gh.get_repo(full).get_pull(num)
    return PRDetails(owner, repo, num, pr.title, pr.body)


def hunk_position_for_target_line(hunk: TypingAny, target_line_number: int) -> int:
    """
    Given a hunk and an absolute target_line_number (new file),
    return the 0-based diff position for that line within this hunk.
    """
    position = 0
    current_target = hunk.target_start
    for line in hunk:
        if line.is_added or line.is_context or line.is_removed:
            if (line.is_added or line.is_context) and current_target == target_line_number:
                return position
            position += 1
            if line.is_added or line.is_context:
                current_target += 1
    raise ValueError(f"Line {target_line_number} not found in hunk starting at {hunk.target_start}")


def analyze_code(parsed: PatchSet, pr_details: PRDetails) -> List[Dict[str, Any]]:
    reviews = []
    for pf in parsed:
        if not pf.path or pf.path == "/dev/null":
            continue
        for hunk in pf:
            prompt = create_prompt(pf.path, hunk, pr_details)
            ai_reviews = get_ai_response(prompt)
            for r in ai_reviews:
                rel = int(r['lineNumber'])
                file_line = hunk.target_start + rel - 1
                try:
                    position = hunk_position_for_target_line(hunk, file_line)
                except ValueError:
                    continue
                reviews.append({
                    'path': pf.path,
                    'position': position,
                    'body': wrap_body(r),
                    'severity': int(r.get('severity', 0)),
                    'type': r.get('type', 'comment')
                })
    reviews.sort(key=lambda x: x['severity'], reverse=True)
    suggestions = [r for r in reviews if r['type']=='suggestion'][:MAX_SUGGESTIONS]
    comments = [r for r in reviews if r['type']=='comment'][:MAX_COMMENTS]
    chosen = suggestions + comments
    chosen.sort(key=lambda x: x['severity'], reverse=True)
    return chosen


def create_prompt(path: str, hunk: TypingAny, pr: PRDetails) -> str:
    content = '\n'.join(hunk.source)
    return f"""
You are a senior code reviewer.
Review the PR changes in `{path}` and identify **critical or subtle issues** in logic, state, correctness, edge cases, and data handling.
Avoid shallow comments like \"consider a try-catch\" or stylistic nitpicks.

Return JSON in the format:
{{
  \"reviews\": [
    {{
      \"lineNumber\": <relative_line>,
      \"reviewComment\": \"<insightful and actionable comment>\",
      \"severity\": <1-5>,  // 1=minor, 5=critical
      \"type\": \"suggestion\" | \"comment\"
    }}
  ]
}}

Guidelines:
- Use \"type\": \"suggestion\" for code changes, \"comment\" for observations.
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
    model = Client.GenerativeModel(os.environ.get('GEMINI_MODEL','gemini-2.0-flash-001'))
    try:
        text = model.generate_content(prompt, generation_config={"max_output_tokens":8192}).text
        clean = text.strip().lstrip('```json').rstrip('```').strip()
        return json.loads(clean).get('reviews', [])
    except Exception:
        return []


def wrap_body(r: Dict[str, Any]) -> str:
    if r.get('type')=='suggestion':
        return f"```suggestion\n{r['reviewComment']}\n```"
    return r['reviewComment']


def create_review_comment(owner: str, repo: str, num: int, reviews: List[Dict[str, Any]]):
    repo_obj = gh.get_repo(f"{owner}/{repo}")
    pr = repo_obj.get_pull(num)
    head_sha = pr.head.sha

    # Post one inline comment per review (use positional args)
    for rev in reviews:
        pr.create_review_comment(
            rev['body'],      # body of the comment
            head_sha,         # commit SHA
            rev['path'],      # file path
            rev['position']   # diff position
        )


def main():
    pr = get_pr_details()
    if os.environ.get('GITHUB_EVENT_NAME')!='issue_comment':
        return
    ev = json.load(open(os.environ['GITHUB_EVENT_PATH']))
    if not ev.get('issue',{}).get('pull_request'):
        return

    repo = gh.get_repo(f"{pr.owner}/{pr.repo}")
    pull = repo.get_pull(pr.pull_number)
    gh_files = pull.get_files()

    reviews = []
    exclude_patterns = [p.strip() for p in os.environ.get('INPUT_EXCLUDE','').split(',') if p.strip()]
    for gh_file in gh_files:
        if not gh_file.patch:
            continue
        header = (
            f"diff --git a/{gh_file.filename} b/{gh_file.filename}\n"
            f"--- a/{gh_file.filename}\n"
            f"+++ b/{gh_file.filename}\n"
        )
        parsed = PatchSet(header + gh_file.patch)
        parsed = [pf for pf in parsed if pf.path == gh_file.filename]
        parsed = [pf for pf in parsed if not any(fnmatch.fnmatch(pf.path, pat) for pat in exclude_patterns)]
        reviews.extend(analyze_code(parsed, pr))

    if reviews:
        create_review_comment(pr.owner, pr.repository, pr.pull_number, reviews)


if __name__=='__main__':
    main()
