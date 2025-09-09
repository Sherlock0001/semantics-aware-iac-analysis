import pandas as pd

# Read the CSV filesd
df1 = pd.read_csv('szz-validation-commit-changes.csv')
df2 = pd.read_csv('szz-validation-commit-changes_bug_inducing_commit.csv')


# Select specific columns from each DataFrame
df1_selected = df1[['Commit_id','Added_lines_fixing_commit']]
df2_selected = df2[['Commit_id', 'Added_lines_bug_inducing_commit']]

# Merge the DataFrames on the 'id' column
merged_df = pd.merge(df1_selected, df2_selected, on='Commit_id')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_file_ansible_defect.csv', index=False)

print("Files merged successfully!")