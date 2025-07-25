import json
import os
from typing import List, Dict, Any
import google.generativeai as Client
from github import Github
import requests
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


def get_diff(owner: str, repo: str, pull_number: int) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}.diff"
    hdr = {'Authorization': f'Bearer {GITHUB_TOKEN}', 'Accept': 'application/vnd.github.v3.diff'}
    resp = requests.get(url, headers=hdr)
    return resp.text if resp.status_code == 200 else ""


def analyze_code(parsed: PatchSet, pr_details: PRDetails) -> List[Dict[str, Any]]:
    reviews = []
    for pf in parsed:
        if not pf.path or pf.path == "/dev/null":
            continue
        for hunk in pf:
            prompt = create_prompt(pf.path, hunk, pr_details)
            ai_reviews = get_ai_response(prompt)
            for r in ai_reviews:
                # calculate file line number and side
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
    # sort by severity
    reviews.sort(key=lambda x: x['severity'], reverse=True)
    # split suggestions and comments
    suggestions = [r for r in reviews if r['type']=='suggestion'][:MAX_SUGGESTIONS]
    comments = [r for r in reviews if r['type']=='comment'][:MAX_COMMENTS]
    chosen = suggestions + comments
    chosen.sort(key=lambda x: x['severity'], reverse=True)
    return chosen


def create_prompt(path: str, hunk: Any, pr: PRDetails) -> str:
    content = '\n'.join(hunk.source)
    return f"""
Review PR changes in {path}:
Provide JSON: {{"reviews":[{{"lineNumber":<relative_line>,"reviewComment":"<text>","severity":<1-5>,"type":"suggestion"|"comment"}}]}}
- 'type': 'suggestion' for fix, 'comment' otherwise
- Wrap suggestions in ```suggestion``` block
- Include 'severity'

PR Title: {pr.title}
PR Desc:
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
    except:
        return []


def wrap_body(r: Dict[str, Any]) -> str:
    if r.get('type')=='suggestion':
        return f"```suggestion\n{r['reviewComment']}\n```"
    return r['reviewComment']


def create_review_comment(owner: str, repo: str, num: int, reviews: List[Dict[str, Any]]):
    pr = gh.get_repo(f"{owner}/{repo}").get_pull(num)
    # pass 'path', 'line', 'side', 'body'
    gh_comments = [{'path': rev['path'], 'line': rev['line'], 'side': rev['side'], 'body': rev['body']} for rev in reviews]
    pr.create_review(body="AI Review", comments=gh_comments, event="COMMENT")


def main():
    pr = get_pr_details()
    if os.environ.get('GITHUB_EVENT_NAME')!='issue_comment':
        return
    ev = json.load(open(os.environ['GITHUB_EVENT_PATH']))
    if not ev.get('issue',{}).get('pull_request'):
        return
    diff = get_diff(pr.owner, pr.repo, pr.pull_number)
    parsed = PatchSet(diff)
    excl = [p.strip() for p in os.environ.get('INPUT_EXCLUDE','').split(',') if p.strip()]
    parsed = [pf for pf in parsed if not any(fnmatch.fnmatch(pf.path, e) for e in excl)]
    reviews = analyze_code(parsed, pr)
    if reviews:
        create_review_comment(pr.owner, pr.repo, pr.pull_number, reviews)

if __name__=='__main__':
    main()
