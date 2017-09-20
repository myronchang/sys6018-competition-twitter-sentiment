library(XML)
library(tm)
library(tidyverse)
library(boot)
library(MASS)
library(nnet)

data <- read_csv("train.csv")

news = VCorpus(DataframeSource(data))

news.tfidf = DocumentTermMatrix(news, control = list(weighting = weightTfIdf))
news.tfidf
news.tfidf[1:5,1:5]
as.matrix(news.tfidf[1:5,1:5])

# there's a lot in the documents that we don't care about. clean up the corpus.
news.clean = tm_map(news, stripWhitespace)                          # remove extra whitespace
news.clean = tm_map(news.clean, removeNumbers)                      # remove numbers
news.clean = tm_map(news.clean, removePunctuation)                  # remove punctuation
news.clean = tm_map(news.clean, content_transformer(tolower))       # ignore case
news.clean = tm_map(news.clean, removeWords, stopwords("english"))  # remove stop words
news.clean = tm_map(news.clean, stemDocument)                       # stem all words

# compare original content of document 1 with cleaned content
news[[1]]$content
news.clean[[1]]$content  # do we care about misspellings resulting from stemming?

# recompute TF-IDF matrix
news.clean.tfidf = DocumentTermMatrix(news.clean, control = list(weighting = weightTfIdf))
news.clean.tfidf

# reinspect the first 5 documents and first 5 terms
news.clean.tfidf[1:5,1:5]
as.matrix(news.clean.tfidf[1:5,1:5])

cleaneddata <- as.data.frame(as.matrix(news.clean.tfidf))
newdata <- cbind(data,cleaneddata)



# Ordinal logistic regression

newdata2 <- newdata[,-2]  # Drop "text" variable which was causing problems with LOOCV
newdata2$sentiment <- as.factor(newdata2$sentiment)   # Convert sentiment to factor to allow ordinal logistic regression to run

# Model and summary
#ordinal.model <- polr(sentiment ~ ., data = newdata2, Hess=TRUE)   # Don't run; crashes R session
summary(ordinal.model)

# Confidence intervals of variables
confint(ordinal.model)

# LOOCV
cvv1 <- sapply(X = 1:nrow(newdata2), function(X){
  training_data <- newdata2[-X,]
  testing_data <- newdata2[X,]
  cvmodel <- polr(sentiment ~ ., data = newdata2, Hess=TRUE)
  guess <- predict(cvmodel, newdata=testing_data)
  accuracy <- guess == testing_data$sentiment
  return(accuracy)
})

totalaccuracy <- sum(cvv1) / length(cvv1)   # 
















# Use model #2 to generate predictions for test data
testing <- read_csv("test.csv")

# Clean up test data same way as training data
test = VCorpus(DataframeSource(testing))

test.tfidf = DocumentTermMatrix(test, control = list(weighting = weightTfIdf))
test.tfidf
test.tfidf[1:5,1:5]
as.matrix(test.tfidf[1:5,1:5])

# there's a lot in the documents that we don't care about. clean up the corpus.
test.clean = tm_map(test, stripWhitespace)                          # remove extra whitespace
test.clean = tm_map(test.clean, removeNumbers)                      # remove numbers
test.clean = tm_map(test.clean, removePunctuation)                  # remove punctuation
test.clean = tm_map(test.clean, content_transformer(tolower))       # ignore case
test.clean = tm_map(test.clean, removeWords, stopwords("english"))  # remove stop words
test.clean = tm_map(test.clean, stemDocument)                       # stem all words

# compare original content of document 1 with cleaned content
test[[1]]$content
test.clean[[1]]$content  # do we care about misspellings resulting from stemming?

# recompute TF-IDF matrix
test.clean.tfidf = DocumentTermMatrix(test.clean, control = list(weighting = weightTfIdf))
test.clean.tfidf

# reinspect the first 5 documents and first 5 terms
test.clean.tfidf[1:5,1:5]
as.matrix(test.clean.tfidf[1:5,1:5])

# we've still got a very sparse document-term matrix. remove sparse terms at various thresholds.
#test.tfidf.99 = removeSparseTerms(test.clean.tfidf, 0.99)  # remove terms that are absent from at least 99% of documents (keep most terms)
#test.tfidf.99
#as.matrix(test.tfidf.99[1:5,1:5])
#as.matrix(test.tfidf.99)
#as.data.frame(test.tfidf.99)

#test.tfidf.98 = removeSparseTerms(test.clean.tfidf, 0.98)  # remove terms that are absent from at least 99% of documents (keep most terms)
#test.tfidf.98
#as.matrix(test.tfidf.98[1:5,1:5])
#as.matrix(test.tfidf.98)

cleanedtestdata <- as.data.frame(as.matrix(test.clean.tfidf))
newtestdata <- cbind(testing,cleanedtestdata)

predictions <- as.data.frame(round(predict(model2, newdata=newtestdata)))
names(predictions) <- "sentiment"
submission <- cbind(newtestdata$id, predictions)
names(submission) <- c("id","sentiment")

write.csv(submission, "ParametricModelPredictions.csv", row.names=FALSE)
