import json
import os
from typing import List, Dict, Any
import google.generativeai as Client
from github import Github
import requests
import fnmatch
from unidiff import Hunk

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
MAX_SUGGESTIONS = 3
MAX_COMMENTS = 2

# Initialize GitHub and Gemini clients
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


def get_diff(owner: str, repo: str, pull_number: int) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}.diff"
    hdr = {'Authorization': f'Bearer {GITHUB_TOKEN}', 'Accept': 'application/vnd.github.v3.diff'}
    resp = requests.get(url, headers=hdr)
    return resp.text if resp.status_code == 200 else ""


def analyze_code(parsed: List[Dict[str, Any]], pr_details: PRDetails) -> List[Dict[str, Any]]:
    all_reviews = []
    for f in parsed:
        path = f.get('path')
        if not path or path == "/dev/null":
            continue
        for h in f.get('hunks', []):
            lines = h.get('lines', [])
            if not lines:
                continue
            hunk = Hunk()
            hunk.source_length = len(lines)
            hunk.content = '\n'.join(lines)
            prompt = create_prompt(path, hunk, pr_details)
            resp = get_ai_response(prompt)
            for r in resp:
                all_reviews.append({
                    'path': path,
                    'position': int(r['lineNumber']),
                    'body': wrap_body(r),
                    'severity': int(r.get('severity', 0)),
                    'type': r.get('type', 'comment')
                })
    all_reviews.sort(key=lambda x: x['severity'], reverse=True)
    suggestions = [r for r in all_reviews if r['type'] == 'suggestion'][:MAX_SUGGESTIONS]
    comments = [r for r in all_reviews if r['type'] == 'comment'][:MAX_COMMENTS]
    chosen = suggestions + comments
    chosen.sort(key=lambda x: x['severity'], reverse=True)
    return chosen


def create_prompt(path: str, hunk: Hunk, pr: PRDetails) -> str:
    return f'''
Your task: review this PR diff.
Instructions:
- Output JSON: {{"reviews":[{{"lineNumber":<line>,"reviewComment":"<text>","severity":<1-5>,"type":"suggestion" or "comment"}}]}}
- "type" must be "suggestion" for fixes, "comment" for remarks.
- Wrap suggestions in ```suggestion``` markdown with replacement code.
- Use GitHub Markdown.
- Focus only on real issues.

PR title: {pr.title}
PR description:
---
{pr.description or 'None'}
---
```diff
{hunk.content}
```
'''


def get_ai_response(prompt: str) -> List[Dict[str, Any]]:
    model = Client.GenerativeModel(os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash-001'))
    try:
        txt = model.generate_content(prompt, generation_config={"max_output_tokens": 8192}).text
        txt = txt.strip().lstrip('```json').rstrip('```').strip()
        return json.loads(txt).get('reviews', [])
    except:
        return []


def wrap_body(r: Dict[str, Any]) -> str:
    if r.get('type') == 'suggestion':
        return f"```suggestion\n{r['reviewComment']}\n```"
    return r['reviewComment']


def create_review_comment(owner: str, repo: str, num: int, reviews: List[Dict[str, Any]]):
    pr = gh.get_repo(f"{owner}/{repo}").get_pull(num)
    pr.create_review(
        body="AI Review",
        comments=[{'path': rev['path'], 'position': rev['position'], 'body': rev['body']} for rev in reviews],
        event="COMMENT"
    )


def parse_diff(diff: str) -> List[Dict[str, Any]]:
    files, cur, h = [], None, None
    for line in diff.splitlines():
        if line.startswith('diff --git'):
            if cur: files.append(cur)
            cur = {'path': '', 'hunks': []}
        elif line.startswith('--- a/') and cur:
            cur['path'] = line[6:]
        elif line.startswith('+++ b/') and cur:
            cur['path'] = line[6:]
        elif line.startswith('@@') and cur:
            h = {'lines': []}
            cur['hunks'].append(h)
        elif h is not None:
            h['lines'].append(line)
    if cur: files.append(cur)
    return files


def main():
    pr = get_pr_details()
    if os.environ.get('GITHUB_EVENT_NAME') != 'issue_comment':
        return
    data = json.load(open(os.environ['GITHUB_EVENT_PATH']))
    if not data.get('issue', {}).get('pull_request'):
        return
    diff = get_diff(pr.owner, pr.repo, pr.pull_number)
    parsed = parse_diff(diff)
    excl = [p.strip() for p in os.environ.get('INPUT_EXCLUDE', '').split(',') if p.strip()]
    parsed = [f for f in parsed if not any(fnmatch.fnmatch(f['path'], e) for e in excl)]
    reviews = analyze_code(parsed, pr)
    if reviews:
        create_review_comment(pr.owner, pr.repo, pr.pull_number, reviews)

if __name__ == '__main__':
    main()
