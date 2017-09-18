# Adrian Mead, Myron Chang, Yi Hao
# SYS class
# Kaggle Competition 3 - Twitter Self-driving Car Sentiment
# September 21, 2017

# Load in the libraries
library(tidyverse)
library(tm)

# Read in the training data
train_data <- read_csv(file = 'train.csv')

# When dealing with text we only need the character column
train_tweets <- as.tibble(train_data$text)

# convert this part of the data frame to a corpus object.
twitter_corpus = VCorpus(DataframeSource(train_tweets))

# compute TF-IDF matrix and inspect sparsity
twitter.tfidf = DocumentTermMatrix(twitter_corpus, control = list(weighting = weightTfIdf))
twitter.tfidf  # non-/sparse entries indicates how many of the DTM cells are non-zero and zero, respectively.
# sparsity is number of non-zero cells divided by number of zero cells.

##### Reducing Term Sparsity #####

# there's a lot in the documents that we don't care about. clean up the corpus.
twitter_corpus.clean = tm_map(twitter_corpus, stripWhitespace)                          # remove extra whitespace
twitter_corpus.clean = tm_map(twitter_corpus.clean, removeNumbers)                      # remove numbers
twitter_corpus.clean = tm_map(twitter_corpus.clean, removePunctuation)                  # remove punctuation
twitter_corpus.clean = tm_map(twitter_corpus.clean, content_transformer(tolower))       # ignore case
# twitter_corpus.clean = tm_map(twitter_corpus.clean, removeWords, stopwords("english"))  # remove stop words
twitter_corpus.clean = tm_map(twitter_corpus.clean, stemDocument)                       # stem all words

twitter_corpus[[1]]$content
twitter_corpus.clean[[1]]$content  # do we care about misspellings resulting from stemming?

# recompute TF-IDF matrix
twitter.clean.tfidf = DocumentTermMatrix(twitter_corpus.clean, control = list(weighting = weightTfIdf))

# reinspect the first 5 documents and first 5 terms
twitter.clean.tfidf[1:5,1:5]
as.matrix(twitter.clean.tfidf[1:5,1:5])

# we've still got a very sparse document-term matrix. remove sparse terms at various thresholds.
twitter.tfidf.99 = removeSparseTerms(twitter.clean.tfidf, .7)  # remove terms that are absent from at least 99% of documents (keep most terms)
twitter.tfidf.99
as.matrix(twitter.tfidf.99[1:5,1:5])
# Gets us to 177 terms

# Let's try out the function with some data
train <- twitter.tfidf.99
test <- 
prediction_column <- train_data$sentiment
k <- 1

#Now going to code up KNN
funcKNN <- function(train, test = NA, prediction_column, k){
  # Perform cross-validation on the full set of data when test is NA
  if(is.na(test)){
    # Default to LOOCV
    mtrx.tfidf = as.matrix(train)
    dist.matrix = as.matrix(dist(mtrx.tfidf))
    # LOOCV is pretty easy; we don't need to do anything random as the process is deterministic
    correct_pred <- sapply(X = 1 : NROW(dist.matrix), function(X){
      most.similar.documents = order(dist.matrix[X,], decreasing = FALSE)
      # Use k to find the k-nearest neighbor
      knn <- most.similar.documents[2:(k+1)]
      prediction_value <- round(mean(prediction_column[knn]))
      return(prediction_value == prediction_column[X])
    })
    accuracy <- sum(correct_pred) / length(correct_pred)
    return(accuracy)
  }
}
funcKNN(twitter.tfidf.99, prediction_column = train_data$sentiment, k = 3)
sapply(X = 1:50, function(X){funcKNN(twitter.tfidf.99, NA, train_data$sentiment, X)})

