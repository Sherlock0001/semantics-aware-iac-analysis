# Load required libraries
library(tm)
library(SnowballC)
library(topicmodels)

# Define extra stop words
extra_stop_words <- c("ensure", "require", "file", "package", "service", "exec", "class", "puppet")

# Read the CSV file containing script content
data <- read.csv("scripts_with_labels_no_commnts.csv", encoding = "latin1", stringsAsFactors = FALSE)

# Check data structure
print(paste("Number of rows in input data:", nrow(data)))
print(paste("Column names:", paste(colnames(data), collapse = ", ")))

# Assuming the script content is in a column named 'Script.Content'
formatted_string_list <- data$Script.Content
docs <- Corpus(VectorSource(formatted_string_list))

# Define preprocessing function
preprocess_text <- function(x) {
  x <- tolower(x)
  x <- gsub("\\s+", " ", x)             # Replace multiple spaces with single space
  x <- gsub("[[:punct:]]", " ", x)      # Replace punctuation with space
  x <- gsub("\\d+", " ", x)             # Replace numbers with space
  x <- gsub("\\s+", " ", x)             # Again replace multiple spaces with single space
  x <- trimws(x)                        # Trim leading and trailing whitespaces
  if (nchar(x) == 0) x <- "emptydocument"  # Replace empty documents with a placeholder
  return(x)
}

# Preprocess the text
docs <- tm_map(docs, content_transformer(preprocess_text))

# Remove standard stop words and extra stop words
docs <- tm_map(docs, removeWords, c(stopwords("english"), extra_stop_words))

# Tokenization and stemming
docs <- tm_map(docs, stemDocument, language = "porter")

# Create a Document-Term Matrix (DTM)
dtm <- DocumentTermMatrix(docs)

# Apply TF-IDF weighting
dtm_tfidf <- weightTfIdf(dtm)

# Remove sparse terms
max_sparsity_allowed <- 0.99
dtm_tfidf <- removeSparseTerms(dtm_tfidf, max_sparsity_allowed)

# Convert DTM to a regular matrix
dtm_matrix <- as.matrix(dtm_tfidf)

# Create a data frame with script index, defect label, and dataset name
metadata <- data.frame(
  ScriptIndex = 1:nrow(data),
  DefectLabel = data$Defect.Label,  # Adjust column name as per your data
  DatasetName = data$Dataset.Name   # Modify dataset name if needed
)

# Combine metadata with DTM
final_data <- cbind(metadata, dtm_matrix)

# Save the final data
write.csv(final_data, "structured_dtm_tfidf.csv", row.names = FALSE)

# Print the first few rows and dimensions of the final data
print(head(final_data))
print(dim(final_data))