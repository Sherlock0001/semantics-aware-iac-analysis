# import csv
# import pickle
# import re
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer

# # Initialize Porter Stemmer
# porter_stemmer = PorterStemmer()

# # Function to remove stop words from a given text
# def remove_stop_words(text):
#     words = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     filtered_words = [word for word in words if word.lower() not in stop_words]
#     return ' '.join(filtered_words)

# # Function to remove comments from a given text
# def remove_comments(text):
#     # Remove single-line comments (e.g., #, //)
#     text = re.sub(r'#.*', '', text)
#     text = re.sub(r'//.*', '', text)
#     # Remove multi-line comments (e.g., /* ... */)
#     text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
#     return text

# # Function to preprocess the script content by removing comments, stop words, and stemming
# def preprocess_script_content(script_content):
#     script_content_no_comments = remove_comments(script_content)
#     script_content_no_stopwords = remove_stop_words(script_content_no_comments)
#     stemmed_words = [porter_stemmer.stem(word) for word in word_tokenize(script_content_no_stopwords)]
#     return ' '.join(stemmed_words)

# # Load the pickled dictionary with utf-8 encoding
# with open('SCRIPT.LABELS.DUMP', 'rb') as file:
#     all_script_dict = pickle.load(file, encoding='utf-8')

# # Define the CSV file path
# output_csv = 'scripts_with_labels.csv'

# # Open the CSV file for writing with utf-8 encoding
# with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
#     fieldnames = ['File ID', 'Script Content', 'Defect Label', 'Dataset Name']
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

#     # Write the header
#     writer.writeheader()

#     # Write each script's details to the CSV file, preprocessing the content
#     for file_id, (script_content, defect_label, dataset_name) in all_script_dict.items():
#         # Check if defect_label is not empty
#         if defect_label.strip():  # Check if label is not empty or whitespace only
#             preprocessed_content = preprocess_script_content(script_content)
#             writer.writerow({
#                 'File ID': file_id,
#                 'Script Content': preprocessed_content,
#                 'Defect Label': defect_label,
#                 'Dataset Name': dataset_name
#             })
#             # Add an empty line as separator between script entries
#             csv_file.write('\n')

# print(f"Scripts have been written to {output_csv} with comments, stop words, and stemming applied.")




import csv
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize Porter Stemmer
porter_stemmer = PorterStemmer()

# Function to remove stop words from a given text
def remove_stop_words(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Function to remove comments from a given text
def remove_comments(text):
    # Remove single-line comments (e.g., #, //)
    text = re.sub(r'#.*', '', text)
    text = re.sub(r'//.*', '', text)
    # Remove multi-line comments (e.g., /* ... */)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text

# Function to preprocess the script content by removing comments, stop words, and other preprocessing steps
def preprocess_script_content(script_content):
    script_content_no_comments = remove_comments(script_content)
    script_content_cleaned = remove_stop_words(script_content_no_comments)
    
    # Remove apostrophes
    script_content_cleaned = script_content_cleaned.replace("'", "")
    
    # Replace punctuation with space
    script_content_cleaned = re.sub(r'[^\w\s]', ' ', script_content_cleaned)
    
    # Convert to lowercase
    script_content_cleaned = script_content_cleaned.lower()
    
    # Remove single or double-character words
    script_content_cleaned = re.sub(r'\b\w{1,2}\b', '',   script_content_cleaned)
    
    return   script_content_cleaned

# Load the pickled dictionary with utf-8 encoding
with open('SCRIPT.LABELS.DUMP', 'rb') as file:
    all_script_dict = pickle.load(file, encoding='utf-8')

# Define the CSV file path
output_csv = 'scripts_with_labels.csv'

# Open the CSV file for writing with utf-8 encoding
with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ['File ID', 'Script Content', 'Defect Label', 'Dataset Name']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write each script's details to the CSV file, preprocessing the content
    for file_id, (script_content, defect_label, dataset_name) in all_script_dict.items():
        # Preprocess script content
        preprocessed_content = preprocess_script_content(script_content)
        
        # Check if any column is null, if so, skip writing this row
        if None in [preprocessed_content, defect_label, dataset_name]:
            continue
        
        writer.writerow({
            'File ID': file_id,
            'Script Content': preprocessed_content,
            'Defect Label': defect_label,
            'Dataset Name': dataset_name
        })

print(f"Scripts have been written to {output_csv} with comments, stop words, and other preprocessing steps applied")
