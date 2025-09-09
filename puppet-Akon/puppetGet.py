


import pickle

# Load the pickle file with the latin1 encoding
with open('SCRIPT.LABELS.DUMP', 'rb') as file:
    all_script_dict = pickle.load(file, encoding='latin1')

# Iterate over the dictionary
for file_id, (script_content, defect_label, dataset_name) in all_script_dict.items():
    # Print file ID, script content, defect label, and dataset name
    print("File ID:", file_id)
    print("Script Content:", script_content)
    print("Defect Label:", defect_label)
    print("Dataset Name:", dataset_name)
    print("\n")





# Load the pickle file with the latin1 encoding

number_of_scripts = len(all_script_dict)

# Print the number of scripts
print("Number of scripts:", number_of_scripts)