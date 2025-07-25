import json
import os
from typing import List, Dict, Any
import google.generativeai as Client
from github import Github
import difflib
import requests
import fnmatch
from unidiff import Hunk, PatchedFile, PatchSet

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
MAX_COMMENTS = 5  # Hard cap on number of suggestions

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
    """Retrieves details of the pull request from GitHub Actions event payload."""
    with open(os.environ["GITHUB_EVENT_PATH"], "r") as f:
        event_data = json.load(f)

    # Handle comment trigger differently from direct PR events
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
    """Fetches the diff of the pull request from GitHub API."""
    repo_name = f"{owner}/{repo}"
    print(f"Attempting to get diff for: {repo_name} PR#{pull_number}")
    repo_obj = gh.get_repo(repo_name)
    pr = repo_obj.get_pull(pull_number)

    api_url = f"https://api.github.com/repos/{repo_name}/pulls/{pull_number}.diff"
    headers = {
        'Authorization': f'Bearer {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3.diff'
    }

    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        print(f"Retrieved diff length: {len(response.text)}")
        return response.text
    else:
        print(f"Failed to get diff. Status: {response.status_code}")
        print(response.text)
        return ""


def analyze_code(parsed_diff: List[Dict[str, Any]], pr_details: PRDetails) -> List[Dict[str, Any]]:
    """Analyzes the code changes using Gemini and generates review comments."""
    print("Starting analyze_code...")
    comments: List[Dict[str, Any]] = []

    for file_data in parsed_diff:
        file_path = file_data.get('path', '')
        if not file_path or file_path == "/dev/null":
            continue

        class FileInfo:
            def __init__(self, path):
                self.path = path

        file_info = FileInfo(file_path)
        hunks = file_data.get('hunks', [])

        for hunk_data in hunks:
            hunk_lines = hunk_data.get('lines', [])
            if not hunk_lines:
                continue

            hunk = Hunk()
            hunk.source_start = 1
            hunk.source_length = len(hunk_lines)
            hunk.target_start = 1
            hunk.target_length = len(hunk_lines)
            hunk.content = '\n'.join(hunk_lines)

            prompt = create_prompt(file_info, hunk, pr_details)
            ai_response = get_ai_response(prompt)

            if ai_response:
                new_comments = create_comment(file_info, hunk, ai_response)
                comments.extend(new_comments)

            # Stop if we reached the maximum
            if len(comments) >= MAX_COMMENTS:
                print(f"Reached max comments ({MAX_COMMENTS}), stopping analysis.")
                return comments[:MAX_COMMENTS]

    return comments[:MAX_COMMENTS]


def create_prompt(file: PatchedFile, hunk: Hunk, pr_details: PRDetails) -> str:
    """Creates the prompt for the Gemini model."""
    return f"""Your task is reviewing pull requests. Instructions:
    - Provide the response in following JSON format:  {{"reviews": [{{"lineNumber":  <line_number>, "reviewComment": "<review comment>"}}]}}
    - Provide comments and suggestions ONLY if there is something to improve, otherwise "reviews" should be an empty array.
    - Use GitHub Markdown in comments
    - **IMPORTANT**: wrap every suggestion in a GitHub suggestion box, e.g.:

      ```suggestion
      // your replacement code here
      ```

    - Focus on bugs, security issues, and performance problems
    - IMPORTANT: NEVER suggest adding comments to the code

Review the following code diff in the file "{file.path}" and take the pull request title and description into account when writing the response.

Pull request title: {pr_details.title}
Pull request description:

---
{pr_details.description or 'No description provided'}
---

Git diff to review:

```diff
{hunk.content}
```"""


def get_ai_response(prompt: str) -> List[Dict[str, str]]:
    """Sends the prompt to Gemini API and retrieves the response."""
    gemini_model = Client.GenerativeModel(os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash-001'))
    generation_config = {"max_output_tokens": 8192, "temperature": 0.8, "top_p": 0.95}

    try:
        response = gemini_model.generate_content(prompt, generation_config=generation_config)
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        data = json.loads(response_text)
        reviews = data.get("reviews", []) if isinstance(data, dict) else []
        valid = [r for r in reviews if "lineNumber" in r and "reviewComment" in r]
        return valid
    except Exception as e:
        print(f"Error in Gemini call or parsing: {e}")
        return []


def create_comment(file: FileInfo, hunk: Hunk, ai_responses: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Creates comment objects from AI responses."""
    comments: List[Dict[str, Any]] = []
    for ai_response in ai_responses:
        try:
            line_number = int(ai_response["lineNumber"])
            # Validate line range
            if line_number < 1 or line_number > hunk.source_length:
                continue

            # Wrap in suggestion syntax
            suggestion_body = f"```suggestion
{ai_response['reviewComment']}
```"
            comment = {"body": suggestion_body, "path": file.path, "position": line_number}
            if len(comments) < MAX_COMMENTS:
                comments.append(comment)
        except Exception:
            continue

    return comments


def create_review_comment(
    owner: str,
    repo: str,
    pull_number: int,
    comments: List[Dict[str, Any]],
):
    """Submits the review comments to the GitHub API."""
    repo_obj = gh.get_repo(f"{owner}/{repo}")
    pr = repo_obj.get_pull(pull_number)
    try:
        pr.create_review(body="Gemini AI Code Reviewer Comments", comments=comments, event="COMMENT")
        print("Review submitted.")
    except Exception as e:
        print(f"Error creating review: {e}")


def parse_diff(diff_str: str) -> List[Dict[str, Any]]:
    """Parses the diff string and returns a structured format."""
    files = []
    current_file = None
    current_hunk = None

    for line in diff_str.splitlines():
        if line.startswith('diff --git'):
            if current_file:
                files.append(current_file)
            current_file = {'path': '', 'hunks': []}
        elif line.startswith('--- a/') and current_file is not None:
            current_file['path'] = line[6:]
        elif line.startswith('+++ b/') and current_file is not None:
            current_file['path'] = line[6:]
        elif line.startswith('@@') and current_file is not None:
            current_hunk = {'header': line, 'lines': []}
            current_file['hunks'].append(current_hunk)
        elif current_hunk is not None:
            current_hunk['lines'].append(line)

    if current_file:
        files.append(current_file)
    return files


def main():
    pr_details = get_pr_details()
    event_data = json.load(open(os.environ["GITHUB_EVENT_PATH"], "r"))
    event_name = os.environ.get("GITHUB_EVENT_NAME")

    if event_name == "issue_comment":
        if not event_data.get("issue", {}).get("pull_request"):
            return

        diff = get_diff(pr_details.owner, pr_details.repo, pr_details.pull_number)
        if not diff:
            return

        parsed = parse_diff(diff)
        exclude_raw = os.environ.get("INPUT_EXCLUDE", "")
        exclude = [p.strip() for p in exclude_raw.split(",") if p.strip()]
        filtered = [f for f in parsed if not any(fnmatch.fnmatch(f['path'], pat) for pat in exclude)]
        comments = analyze_code(filtered, pr_details)
        if comments:
            create_review_comment(pr_details.owner, pr_details.repo, pr_details.pull_number, comments)
    else:
        print("Unsupported event.")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"Error: {err}")
