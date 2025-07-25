import json
import os
from typing import List, Dict, Any
import re
import google.generativeai as Client
from github import Github
import requests
import fnmatch
from unidiff import Hunk, PatchedFile, PatchSet

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
MAX_COMMENTS = 5  # Hard cap on total review items
SUGGESTION_COUNT = 3  # Number of top items as suggestions

# Initialize GitHub and Gemini clients
gh = Github(GITHUB_TOKEN)
gemini_client = Client.configure(api_key=os.environ.get('GEMINI_API_KEY'))

class PRDetails:
    def __init__(self, owner: str, repo: str, pull_number: int, title: str, description: str):
        self.owner = owner
        self.repo = repo
        self.pull_number = pull_number
        self.title = title
        self.description = description


def get_pr_details() -> PRDetails:
    with open(os.environ["GITHUB_EVENT_PATH"], "r") as f:
        event_data = json.load(f)
    if "issue" in event_data and "pull_request" in event_data["issue"]:
        pull_number = event_data["issue"]["number"]
        repo_full_name = event_data["repository"]["full_name"]
    else:
        pull_number = event_data["number"]
        repo_full_name = event_data["repository"]["full_name"]
    owner, repo = repo_full_name.split("/")
    repo_obj = gh.get_repo(repo_full_name)
    pr = repo_obj.get_pull(pull_number)
    return PRDetails(owner, repo_obj.name, pull_number, pr.title, pr.body)


def get_diff(owner: str, repo: str, pull_number: int) -> str:
    repo_name = f"{owner}/{repo}"
    api_url = f"https://api.github.com/repos/{repo_name}/pulls/{pull_number}.diff"
    headers = {'Authorization': f'Bearer {GITHUB_TOKEN}', 'Accept': 'application/vnd.github.v3.diff'}
    response = requests.get(api_url, headers=headers)
    return response.text if response.status_code == 200 else ""


def analyze_code(parsed_diff: List[Dict[str, Any]], pr_details: PRDetails) -> List[Dict[str, Any]]:
    comments: List[Dict[str, Any]] = []
    for file_data in parsed_diff:
        path = file_data.get('path', '')
        if not path or path == "/dev/null":
            continue
        class FileInfo:
            def __init__(self, path): self.path = path
        file_info = FileInfo(path)
        for hdata in file_data.get('hunks', []):
            lines = hdata.get('lines', [])
            if not lines:
                continue
            header = hdata.get('header', '')
            m = re.match(r"@@ -\d+(?:,\d+)? \+(?P<new_start>\d+)(?:,(?P<new_len>\d+))? @@", header)
            new_start = int(m.group('new_start')) if m else 1
            new_len = int(m.group('new_len')) if m and m.group('new_len') else len(lines)
            hunk = Hunk()
            hunk.source_start = 1
            hunk.source_length = len(lines)
            hunk.target_start = new_start
            hunk.target_length = new_len
            hunk.content = '\n'.join(lines)
            prompt = create_prompt(file_info, hunk, pr_details)
            ai_responses = get_ai_response(prompt)
            comments.extend(create_comment(file_info, hunk, ai_responses))
    severity_map = lambda c: c.get('severity', 0)
    sorted_comments = sorted(comments, key=severity_map, reverse=True)
    # Take top MAX_COMMENTS
    return sorted_comments[:MAX_COMMENTS]


def create_prompt(file: PatchedFile, hunk: Hunk, pr_details: PRDetails) -> str:
    return f"""Your task is reviewing pull requests. Instructions:
- Provide JSON: {{"reviews":[{{"lineNumber":<line_number>,"reviewComment":"<review comment>","severity":<1-5>}}]}}
- Return empty array if no issues.
- Include severity (1=low to 5=critical).
- Use GitHub Markdown.
- Wrap only suggestions in ```suggestion``` blocks; comments are plain markdown.
- Focus on bugs, security, performance.
- Do NOT suggest adding code comments.

PR title: {pr_details.title}
PR description:
---
{pr_details.description or 'No description provided'}
---
```diff
{hunk.content}
```"""


def get_ai_response(prompt: str) -> List[Dict[str, Any]]:
    model = Client.GenerativeModel(os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash-001'))
    cfg = {"max_output_tokens":8192, "temperature":0.8, "top_p":0.95}
    try:
        resp = model.generate_content(prompt, generation_config=cfg)
        text = resp.text.strip().lstrip('```json').rstrip('```').strip()
        data = json.loads(text)
        return [r for r in data.get('reviews', []) if 'lineNumber' in r and 'reviewComment' in r]
    except:
        return []


def create_comment(file: Any, hunk: Hunk, ai_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    comments = []
    for r in ai_responses:
        try:
            ln = int(r['lineNumber'])
            if not (1 <= ln <= hunk.source_length):
                continue
            position = hunk.target_start + ln - 1
            body_text = r['reviewComment']
            sev = int(r.get('severity', 0))
            comments.append({'path': file.path, 'position': position, 'body_text': body_text, 'severity': sev})
        except:
            continue
    return comments


def create_review_comment(owner: str, repo: str, pr_num: int, comments: List[Dict[str, Any]]):
    # First SUGGESTION_COUNT as suggestions, rest as plain comments
    pr = gh.get_repo(f"{owner}/{repo}").get_pull(pr_num)
    payload = []
    for idx, c in enumerate(comments):
        if idx < SUGGESTION_COUNT:
            body = f"```suggestion
{c['body_text']}
```"
        else:
            body = c['body_text']
        payload.append({'path': c['path'], 'position': c['position'], 'body': body})
    pr.create_review(body="AI Review Comments", comments=payload, event="COMMENT")


def parse_diff(diff_str: str) -> List[Dict[str, Any]]:
    files, current, hunk = [], None, None
    for line in diff_str.splitlines():
        if line.startswith('diff --git'):
            if current: files.append(current)
            current = {'path': '', 'hunks': []}
            hunk = None
        elif line.startswith('--- a/') and current:
            current['path'] = line[6:]
        elif line.startswith('+++ b/') and current:
            current['path'] = line[6:]
        elif line.startswith('@@') and current:
            hunk = {'header': line, 'lines': []}
            current['hunks'].append(hunk)
        elif hunk is not None:
            hunk['lines'].append(line)
    if current: files.append(current)
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
    filtered = [f for f in parsed if not any(fnmatch.fnmatch(f['path'], pat) for pat in excl)]
    comments = analyze_code(filtered, pr)
    if comments:
        create_review_comment(pr.owner, pr.repo, pr.pull_number, comments)

if __name__ == '__main__':
    main()
