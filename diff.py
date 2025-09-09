
import os
import requests
import re
import csv
import time

# Access GitHub Personal Access Token from environment variable
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')

# Path to CSV file containing repository, commit IDs, and file paths
CSV_FILE = 'ansible.csv'

# Path to the output CSV file for changes
CHANGES_FILE = 'commit_changes.csv'

def fetch_commit_details(owner, repo, ref):
    """Fetch details of a specific commit from GitHub."""
    url = f'https://api.github.com/repos/{owner}/{repo}/commits/{ref}'
    headers = {'Authorization': f'Bearer {ACCESS_TOKEN}'}

    # fetch commit details with headers
    response = requests.get(url, headers=headers)
   
    if response.status_code == 200:
        commit_details = response.json()
        return commit_details
    else:
        print(f"Error: Could not fetch commit details for '{ref}' in repository '{owner}/{repo}'")
        return None

def process_diff(diff_text):
    try:
        added_lines = []
        deleted_lines = []

        # Split the patch text into hunks
        hunks = re.split(r'(?m)^@@', diff_text)

        # Loop 
        for hunk in hunks:
            if not hunk.strip():
                continue

            # Split the hunk into lines
            lines = hunk.split('\n')

            # Extract the hunk header and skip it
            hunk_header = lines.pop(0)

            # Loop 
            for line in lines:
                # Checking if the line starts with '!' 
                if line.startswith('!'):
                    continue
                # Check if the line starts with '-'  deleted line
                elif line.lstrip().startswith('-'):
                    # Append the deleted line to the deleted_lines list
                    deleted_lines.append(line.lstrip()[1:].rstrip('\r'))
                # Check if the line starts with '+' indicating an added line
                elif line.lstrip().startswith('+'):
                    # Append the added line to the added_lines list
                    added_lines.append(line.lstrip()[1:].rstrip('\r'))

        # Return the lists of added and deleted lines
        return added_lines, deleted_lines

    except Exception as e:
        print(f"Error processing diff: {e}")
        return None, None  # Return None if there are parsing issues
def main():
    added_lines_count = 0
    deleted_lines_count = 0

    with open(CHANGES_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Commit ID", "Repo Name", "File Path", "Deleted Lines", "Added Lines"])

    # Read repository, commit IDs, and file paths from CSV
    with open(CSV_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)  # Use DictReader to access columns by name
        for row in reader:
            repo = row.get('repository')  
            commit_id = row.get('commit') 
            file_path = row.get('filepath')  #  'filepath' column

            if '/' not in repo:
                print(f"Error: Invalid repository format '{repo}'. It should be in the format 'owner/repo'.")
                continue
            
            owner, repo_name = repo.split('/')  
            if owner and repo_name and commit_id and file_path:
                commit_details = fetch_commit_details(owner, repo_name, commit_id)
                if commit_details:
                    files = commit_details.get('files', [])
                    for file_info in files:
                        filename = file_info.get('filename')
                        if filename == file_path:
                            patch = file_info.get('patch')
                            added_lines, removed_lines = process_diff(patch)
                            added_lines_count += len(added_lines)
                            deleted_lines_count += len(removed_lines)
                      
                            with open(CHANGES_FILE, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                if removed_lines is not None:
                                    removed_lines_text = "\n".join(removed_lines)
                                else:
                                    removed_lines_text = ""
                                if added_lines is not None:
                                    added_lines_text = "\n".join(added_lines)
                                else:
                                    added_lines_text = ""
                                writer.writerow([commit_id, repo, file_path, removed_lines_text, added_lines_text])
                            
                            break  # Once found, no need to continue searching for the file
                    else:
                        print(f"No '{file_path}' file found for commit '{commit_id}' in repository '{repo}'.")
                else:
                    print(f"Failed to fetch commit details for commit '{commit_id}' in repository '{repo}'.")
            else:
                print("Warning: Insufficient data in CSV row. Skipping.")
            
            # Wait for 1000ms (1 second)
            time.sleep(1)

    print(f"Total added lines: {added_lines_count}")
    print(f"Total deleted lines: {deleted_lines_count}")

if __name__ == "__main__":
    main()
    
