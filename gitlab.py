
    
#     # Do something with the issues, for example, print them
#     for issue in issues:
#         project_id = issue.get('id', 'Project ID not found')
#         print( project_id)
# else:
#     print('Failed to fetch issues:', response.status_code)

import requests
import difflib
import csv
import os
import re
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_md") 


def is_yaml_file(new_path):

    # Split the path on the dot and get the last element as the extension
    extension = new_path.split('.')[-1].lower()
    
    # Check if the extension is 'yml' or 'yaml'
    return extension in ['yml', 'yaml']


def process_diff(diff_text, project_id, commit_id):
    """
    Processes a git diff patch to extract changes and includes project and commit IDs.
    
    Args:
    - diff_text (str): The git diff patch text.
    - project_id (str): The ID of the project.
    - commit_id (str): The ID of the commit.
    
    Returns:
    - Tuple of lists containing before and after changes along with IDs.
    """
    block_pattern = re.compile(r'^@@ -\d+,\d+ \+\d+,\d+ @@.*$')
    changes = {'before': [], 'after': []}
    
    for line in diff_text.split('\n'):
        if block_pattern.match(line):
            continue  # Skip the block header lines
        elif line.startswith('-'):
            changes['before'].append(line[1:])  # Capture and remove the '-' prefix
        elif line.startswith('+'):
            changes['after'].append(line[1:])  # Capture and remove the '+' prefix

    # Combine changes into a single string for before and after
    before_text = "\n".join(changes['before'])
    after_text = "\n".join(changes['after'])

    return project_id, commit_id, before_text, after_text
def write_changes_to_csv(project_id,Title, commit_id, before_changes, after_changes, csv_file_path="diff_changes_nl3.csv"):
    
    print(len(before_changes))
   
    """
    Appends processed changes to a CSV file, including project and commit IDs, ensuring data is appended without overwriting.

    Args:
    - project_id (str): Project ID associated with the changes.
    - commit_id (str): Commit ID associated with the changes.
    - before_changes (list of str): All "before" changes.
    - after_changes (list of str): All "after" changes.
    - csv_file_path (str): Path to the CSV file for appending the changes.
    """
    # Check if the CSV file exists and if a header is needed
    file_exists = os.path.isfile(csv_file_path)
    
    # Open the file in append mode, ensuring data is added to the end of the file
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(["ProjectID","TitleCommit", "CommitID", "Before", "After"])
        # Append the change data as a new row
        if(len(before_changes)>0 and len(after_changes)>0):
           writer.writerow([project_id,Title,commit_id, " ".join(before_changes), " ".join(after_changes)])

           print(f"Changes have been appended to {csv_file_path}.")
def process_diff_for_yaml(diff_text,Title_commit,new_path,project_id,commit_id):
    """
    Processes the diff text to separate lines before and after the change,
    but only if the new_path has a .yml or .yaml extension.

    Args:
    diff_text (str): The diff text to process.
    new_path (str): The path of the new file, to check the extension.

    Returns:
    dict: A dictionary containing the context, lines before the change, lines after the change, and subsequent text.
    """
    # Initialize result dictionary
    result = {}
    print(new_path)
    # Check if new_path ends with a YAML extension
    if is_yaml_file(new_path):
             
        # write_to_csv(project_id,commit_id,before_change,after_change)
        project_id, commit_id, before_changes, after_changes = process_diff(diff_text, project_id, commit_id)
        write_changes_to_csv(project_id,Title_commit,commit_id, before_changes, after_changes)

    else:
        result["error"] = "Provided path does not have a .yml or .yaml extension."

    return result

# def contains_similar_terms(commit_message, keywords, cutoff=0.8):
#     """
#     Check if the commit message contains terms that are similar to the specified keywords.

#     Args:
#     commit_message (str): The commit message to check.
#     keywords (list): A list of keywords related to security.
#     cutoff (float): The similarity threshold. Words with a similarity score above this cutoff will be considered a match.

#     Returns:
#     bool: True if a similar word is found, False otherwise.
#     """
#     # Split the commit message into words and normalize to lowercase
#     words = commit_message.lower().split()

#     # For each word in the commit message, check if there's a similar keyword
#     for word in words:
#         matches = difflib.get_close_matches(word, keywords, n=1, cutoff=cutoff)
#         if matches:  # If there's at least one match, the word is similar enough to one of the keywords
#             return True
#     return False
# def contains_similar_terms(commit_message, keywords, cutoff=0.8):
#     """
#     Check if the commit message contains terms that are semantically similar to the specified keywords.

#     Args:
#     commit_message (str): The commit message to check.
#     keywords (list): A list of keywords related to security.
#     cutoff (float): The similarity threshold. Words with a similarity score above this cutoff will be considered a match.

#     Returns:
#     bool: True if a similar word is found, False otherwise.
#     """
#     # Create a spaCy document for the commit message
#     doc = nlp(commit_message.lower())
    
#     # Create a spaCy document for each keyword
#     keyword_docs = [nlp(keyword.lower()) for keyword in keywords]

#     # Check each token in the commit message for semantic similarity with any of the keywords
#     for token in doc:
#         for keyword_doc in keyword_docs:
#             # spaCy calculates similarity based on the model's word embeddings
#             if token.similarity(keyword_doc) >= cutoff:
#                 return True
#     return False


def contains_similar_terms(commit_message, keywords, cutoff=0.8):

   
    doc = nlp(commit_message.lower())
    
    #  spaCy document for each keyword
    keyword_docs = [nlp(keyword.lower()) for keyword in keywords]

    # Filter tokens by part of speech to focus on nouns and verbs
    relevant_pos = {'NOUN', 'VERB'}
    filtered_tokens = [token for token in doc if token.pos_ in relevant_pos]

    # Check each filtered token in the commit message for semantic similarity with any of the keywords
    for token in filtered_tokens:
        for keyword_doc in keyword_docs:
        
            if token.similarity(keyword_doc) >= cutoff:
                return True
    return False

def fetch_commit_diff(project_id,Title_commit, commit_id, personal_access_token):
 
    diff_url = f'https://gitlab.com/api/v4/projects/{project_id}/repository/commits/{commit_id}/diff'
    headers = {'Authorization': f'Bearer {personal_access_token}'}
    response = requests.get(diff_url, headers=headers)

    if response.status_code == 200:
        diffs = response.json()
        # print(f"Diff for Commit ID {commit_id} in Project ID {project_id}:")
        for diff in diffs:
            process_diff_for_yaml(diff.get('diff'),Title_commit, diff.get('new_path'),project_id,commit_id)
            # print(diff.get('diff'))  # Print the diff content
    else:
        print(f"Failed to fetch diff for commit ID {commit_id} in project ID {project_id}: {response.status_code}")



def fetch_commits_for_project(project_id, personal_access_token):
  
    # Replace 'gitlab.com' with 'gitlab.example.com' if you are using a self-hosted GitLab
    commits_url = f'https://gitlab.com/api/v4/projects/{project_id}/repository/commits'
    headers = {'Authorization': f'Bearer {personal_access_token}'}
    response = requests.get(commits_url, headers=headers)
    security_keywords = [
    "fix", "bug", "bugfix", "error", "secure", "security", "maintain", "maintenance",
    "refactor", "crash", "leak", "attack", "authenticate", "authentication",
    "authorize", "authorization", "cipher", "crack", "decrypt", "encrypt",
    "vulnerable", "vulnerability", "minimize", "optimize"
]
    if response.status_code == 200:
        commits = response.json()
        # print(f" {commits}+ test")
        for commit in commits:
            commit_id = commit.get('id')
            commit_Title = commit.get('title')
            if contains_similar_terms(commit_Title, security_keywords):
               fetch_commit_diff(project_id,commit_Title , commit_id, personal_access_token) #check it the result is depnding on this line 
               print(f"Project ID {project_id} has {len(commits)} commits.{commit_id}")
            else:
                print(f"Failed to fetch commits for project ID not matches with the security keywords {project_id}: {response.status_code}")

# Access Token (replace with your actual token)
# personal_access_token = 'glpat-obncyCJmDpXyCxz7QTPa'

# # The API endpoint for searching projects with 'ansible' in their name
# url = 'https://gitlab.com/api/v4/search?scope=projects&search=ansible+playbook&page=1&per_page=100'  # Adjusted for example

# # The headers for authentication
# headers = {
#     'Authorization': f'Bearer {personal_access_token}'
# }


# response = requests.get(url, headers=headers)


# if response.status_code == 200:
 
#     projects = response.json()
#     print()

#     # Iterate over each project and fetch commits
#     for project in projects:
#         project_id = project.get('id')
#         print(f"Fetching commits for Project ID: {project_id}")
#         fetch_commits_for_project(project_id, personal_access_token)
# else:
#     print('Failed to fetch projects:', response.status_code)



def fetch_all_projects(base_url, search_query, token):
    page = 1
    x_total_pages = 1  # Will be updated from the API response
    
    while True:
        print(f"page_test{page}")
        # Before making a request, check if we already know the total pages and if we've exceeded it
      

        url = f"{base_url}?scope=projects&search={search_query}&page={page}&per_page=100"
        response = requests.get(url, headers={'Authorization': f'Bearer {token}'})

        if response.status_code == 200:
            # Dynamically update the total number of pages with each response
            x_total_pages = int(response.headers.get('X-Total-Pages', 0))

            projects = response.json()
            if projects:
                for project in projects:
                    project_id = project.get('id')
                    # print(f"Fetching commits for Project ID: {project_id}")
                    fetch_commits_for_project(project_id, token)

          
            else:
                
                page += 1  # Increment to fetch the next page
                print(f" no project")
                break
            page += 1  # Increment to fetch the next page
            print(f"{page} fetching page number")
        else:
            print(f"Failed to fetch projects on page {page}: {response.status_code}")
            break
        if page>=x_total_pages:
            break  # Stop if there are no more pages to fetch

# Variables (replace these with your actual values)
personal_access_token = 'glpat-obncyCJmDpXyCxz7QTPa'
base_url = 'https://gitlab.com/api/v4/search'
search_query = 'ansible playbook'

# Start fetching projects
fetch_all_projects(base_url, search_query, personal_access_token)