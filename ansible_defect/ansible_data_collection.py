import os
import requests
import re
import csv

import time


ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')

# Path to CSV file containing repository, commit id, file path, and decision
CSV_FILE = 'szz-validation.csv'

# Path to the output csv
CHANGES_FILE = 'szz-validation-commit-changes_bug_inducing_commit.csv'

def fetch_commit_details(owner, repo_name, ref,file_path,fixing_commit):
    """Fetch details of a specific commit from GitHub and process them."""
    url = f'https://api.github.com/repos/{owner}/{repo_name}/commits/{ref}'
    headers = {'Authorization': f'Bearer {ACCESS_TOKEN}'}


    # fetch the commit  from GitHub API
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        commit_details = response.json()
        # Process the commit details
        files = commit_details.get('files', [])
        for file_info in files:
            filename = file_info.get('filename')
            patch = file_info.get('patch')
            if filename.lstrip() == file_path.lstrip():
                if patch is not None:  # Check if patch exists
                    added_lines, removed_lines = process_diff(patch)
            
                    if removed_lines is not None:
                        removed_lines_text = "\n".join(removed_lines)
                    else:
                        removed_lines_text = ""

                    if added_lines is not None:
                        added_lines_text = "\n".join(added_lines)
                    else:
                        added_lines_text = ""
                

                    yield (ref, repo_name, filename, removed_lines_text, added_lines_text,fixing_commit)

    else:
        print(f"Error: Could not fetch commit details for '{ref}' in repository '{owner}/{repo_name}'")

def process_diff(diff_text):
    try:
        added_lines = []
        deleted_lines = []

        # Split the patch text into hunks
        hunks = re.split(r'(?m)^@@', diff_text)

        # Loop through hunks
        for hunk in hunks:
            if not hunk.strip():
                continue

            # Split the hunk into lines
            lines = hunk.split('\n')

            # Extract the hunk header and skip it
            hunk_header = lines.pop(0)

            # Loop through lines
            for line in lines:
                # Checking if the line starts with '!'
                if line.startswith('!'):
                    continue
                # Check if the line starts with '-' (deleted line)
                elif line.lstrip().startswith('-'):
                    # Append the deleted line to the deleted_lines list
                    deleted_lines.append(line.lstrip()[1:].rstrip('\r'))
                # Check if the line starts with '+' (added line)
                elif line.lstrip().startswith('+'):
                    # Append the added line to the added_lines list
                    added_lines.append(line.lstrip()[1:].rstrip('\r'))

        # Return the lists of added and deleted lines
        return added_lines, deleted_lines

    except Exception as e:
        print(f"Error processing diff: {e}")
        return None, None  # Return None if there are parsing issues


def process_commit(repo, commit_id,fixing_commit, file_path, decision):
    if '/' not in repo:
        print(f"Error: Invalid repository format '{repo}'. It should be in the format 'owner/repo'.")
        return

    owner, repo_name = repo.split('/')
    commit_details = fetch_commit_details(owner, repo_name, commit_id,file_path,fixing_commit)
 
    if commit_details:
        for row in commit_details:
                with open(CHANGES_FILE, 'a', newline='') as csvfile:

                  if decision== "TRUE" :
                    writer = csv.writer(csvfile)
                    writer.writerow(row)
    else:
        print(f"Failed to fetch commit details for commit '{commit_id}' in repository '{repo}'.")

def main():
    with open(CHANGES_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Commit_id_bug", "Repo_name", "File_path", "Deleted_lines_bug_inducing_commit", "Added_lines_bug_inducing_commit","Commit_id"])

    # Read repository, commit id, and file paths from CSV
    rows = []
    with open(CSV_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            repo = row.get('repo')
            bug_inducing_commit = row.get('bug_inducing_commit')
            fixing_commit = row.get('fixing_commit')
            file_path = row.get('filepath')
            decision = row.get('Final decision')
            time.sleep(0.2)
            process_commit(repo, bug_inducing_commit ,fixing_commit, file_path, decision)


    print("Processing complete.")

if __name__ == "__main__":
    main()
