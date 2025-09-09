

# Load the required library
library(stringr)
library(SnowballC)

# Read the CSV file with Latin-1 encoding
data <- read.csv("scripts_with_labels_no_commnts.csv", encoding = "latin1", stringsAsFactors = FALSE)

# Function to remove comments
remove_comments <- function(text) {
  # Remove single-line comments starting with #
  text <- gsub("#.*$", "", text)
  # Remove multi-line comments
  text <- gsub("/\\*(.*?)\\*/", "", text, perl = TRUE)
  return(text)
}

# Pre-processing:
# data$Script.Content <- iconv(data$Script.Content, "latin1", "UTF-8", sub="")
data$Script.Content <- gsub("'", "", data$Script.Content)  # remove apostrophes
# data$Script.Content <- gsub("[[:punct:]]", " ", data$Script.Content)  # replace punctuation with space
data$Script.Content <- gsub("[[:cntrl:]]", " ", data$Script.Content)  # replace control characters with space
data$Script.Content <- str_trim(data$Script.Content)  # remove leading and trailing whitespace
data$Script.Content <- tolower(data$Script.Content)  # force to lowercase
data$Script.Content <- gsub("\\b\\w{1,2}\\b", "", data$Script.Content)  # remove short words
data$Script.Content <- iconv(data$Script.Content, "latin1", "UTF-8", sub="")
# Tokenize the text on space and then collapse the lists into a single string
data$Script.Content <- sapply(data$Script.Content, function(x) paste(unlist(strsplit(x, "\\s+")), collapse = " "))

# Apply stemming
data$Script.Content <- wordStem(data$Script.Content, language = "porter")

# Write the preprocessed content to a new CSV file
output_csv <- "scripts_with_labels_preprocessed.csv"
write.csv(data, file = output_csv, row.names = FALSE)

cat(paste("Preprocessed data has been written to", output_csv, "\n"))
