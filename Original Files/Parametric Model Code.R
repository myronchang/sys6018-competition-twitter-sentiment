library(XML)
library(tm)
library(tidyverse)
library(boot)

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

# we've still got a very sparse document-term matrix. remove sparse terms at various thresholds.
tfidf.99 = removeSparseTerms(news.clean.tfidf, 0.99)  # remove terms that are absent from at least 99% of documents (keep most terms)
tfidf.99
as.matrix(tfidf.99[1:5,1:5])
as.matrix(tfidf.99)
#as.data.frame(tfidf.99)

tfidf.98 = removeSparseTerms(news.clean.tfidf, 0.98)  # remove terms that are absent from at least 99% of documents (keep most terms)
tfidf.98
as.matrix(tfidf.98[1:5,1:5])
as.matrix(tfidf.98)
cleaneddata <- as.data.frame(as.matrix(tfidf.98))
newdata <- cbind(data,cleaneddata)
#as.data.frame(tfidf.98)

# Removes too many
# tfidf.70 = removeSparseTerms(news.clean.tfidf, 0.70)  # remove terms that are absent from at least 70% of documents
# tfidf.70
# as.matrix(tfidf.70[1:5, 1:5])
# news.clean[[1]]$content

# # which documents are most similar?
# dtm.tfidf.99 = as.matrix(tfidf.99)
# dtm.dist.matrix = as.matrix(dist(dtm.tfidf.99))
# most.similar.documents = order(dtm.dist.matrix[1,], decreasing = FALSE)
# news[[most.similar.documents[1]]]$content
# news[[most.similar.documents[2]]]$content
# news[[most.similar.documents[3]]]$content
# news[[most.similar.documents[4]]]$content
# news[[most.similar.documents[5]]]$content


# Add correlation matrix among 43 terms

# Logistic regression (all 43 variables)

newdata2 <- newdata[,-2]  # Drop "text" variable which was causing problems with LOOCV
model <- glm(sentiment~., data=newdata2)
summary(model)
anova(model)

# Leave One Out Cross-validation
cv.err=cv.glm(newdata2, model)
cv.err$delta  # 0.6212805 0.6212479

# Logistic regression #2 (only variables with significance markers '.'s or '*'s from first model)

model2 <- glm(sentiment~cant+come+googl+less+need+think+use+wait+want, data=newdata2)
summary(model2)
anova(model2)

# LOOCV for model #2
cv.err2 = cv.glm(newdata2, model2)
cv.err2$delta # 0.5923352 0.5923267

# Logistic regression #3 (only variables with *s from model 2)

model3 <- glm(sentiment~cant+googl+need+think+use+wait+want, data=newdata2)
summary(model3)
anova(model3)

# LOOCV for model #3
cv.err3 = cv.glm(newdata2, model3)
cv.err3$delta # 0.5925687 0.5925625

# Logistic regression #4 (only variables with ** and *** from model 3)

model4 <- glm(sentiment~cant+googl+want, data=newdata2)
summary(model4)
anova(model4)

# LOOCV for model #4
cv.err4 = cv.glm(newdata2, model4)
cv.err4$delta # 0.5972975 0.5972949

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
