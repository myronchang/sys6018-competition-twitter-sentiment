#Kaggle Competition 3-7:
#Twitter Sentiment

#Myron Chang
#Adrian Mead
#Yi Hao

install.packages("XML", repos = "https://cran.r-project.org/")
install.packages("tm", repos = "https://cran.r-project.org/")
install.packages("SnowballC")
library(XML)
library(tm)
library(tidyverse)
library(SnowballC)

## Import data

train_data <- read.csv("train.csv", header=T, stringsAsFactors = FALSE)
str(train_data)
#train_data$sentiment <- as.factor(train_data$sentiment)

##summarize the two variables
summary(train_data)
nrow(train_data)
prop.table(table(train_data$sentiment))
#         1          2          3          4          5 
#   0.02344546 0.11926606 0.61467890 0.18246687 0.06014271 

## Prepare train_m, test1_m and test2_m for split testing
#s1 <- sample(1:981, size=491) 
#train_m <- train_data[s1,]
#test0 <- train_data[-s1,]
#s2 <- sample(1:490, size=245) 
#test1_m <- test0[s2,]
#test2_m <- test0[-s2,]

#Check the partition results (if it is randomly partitioning)
#nrow(train_m)
#summary(train_m)
#nrow(test1_m)
#summary(test1_m)
#nrow(test2_m)
#summary(test2_m)
#prop.table(table(train_m$sentiment))
#prop.table(table(test1_m$sentiment))
#prop.table(table(test2_m$sentiment))


# Convert the data frame to a corpus object.
train_text_df<-as.data.frame(train_data$text, stringsAsFactors = FALSE)
train_text_corpus = VCorpus(DataframeSource(train_text_df))

# regular indexing returns a sub-corpus
inspect(train_text_corpus[1:2])

# double indexing accesses actual documents
train_text_corpus[[1]]
train_text_corpus[[1]]$content

# Compute TF-IDF matrix and inspect sparsity
train_text.tfidf = DocumentTermMatrix(train_text_corpus, control = list(weighting = weightTfIdf))
train_text.tfidf  # non-/sparse entries indicates how many of the DTM cells are non-zero and zero, respectively.
# sparsity is number of non-zero cells divided by number of zero cells.

# inspect sub-matrix:  first 5 documents and first 5 terms
train_text.tfidf[1:5,1:5]
as.matrix(train_text.tfidf[1:5,1:5])

## Reducing Term Sparsity

# there's a lot in the documents that we don't care about. clean up the corpus.
train_text.clean = tm_map(train_text_corpus, stripWhitespace)                   # remove extra whitespace
train_text.clean = tm_map(train_text.clean, removeNumbers)                      # remove numbers
train_text.clean = tm_map(train_text.clean, removePunctuation)                  # remove punctuation
train_text.clean = tm_map(train_text.clean, content_transformer(tolower))       # ignore case
train_text.clean = tm_map(train_text.clean, removeWords, stopwords("english"))  # remove stop words
train_text.clean = tm_map(train_text.clean, stemDocument, language = "english") # stem all words

# compare original content of document 1 with cleaned content
train_text_corpus[[1]]$content
train_text.clean[[1]]$content  # do we care about misspellings resulting from stemming?

# recompute TF-IDF matrix
train_text.clean.tfidf = DocumentTermMatrix(train_text.clean, control = list(weighting = weightTfIdf))

# reinspect the first 5 documents and first 5 terms
train_text.clean.tfidf[1:5,1:5]
as.matrix(train_text.clean.tfidf[1:5,1:5])

# we've still got a very sparse document-term matrix. remove sparse terms at various thresholds.
train.tfidf.99 = removeSparseTerms(train_text.clean.tfidf, 0.99)  # remove terms that are absent from at least 99% of documents (keep most terms)
train.tfidf.99
as.matrix(train.tfidf.99[1:5,1:5])
clean_train99 <- as.data.frame(as.matrix(train.tfidf.99))
#sentiment <- train.data$sentiment
#clean.train99 <- cbind(sentiment,clean_train99)

train.tfidf.98 = removeSparseTerms(train_text.clean.tfidf, 0.98)  # remove terms that are absent from at least 98% of documents (keep most terms)
train.tfidf.98
as.matrix(train.tfidf.98[1:5,1:5])
clean_train98 <- as.data.frame(as.matrix(train.tfidf.98))

##########################################################################
## II. Fitting Regression and classification Models and Prediction

###(1) Have the training data train_data98_m ready:

#transform the format from Documetn Term Matrix to Matrix
train98_BoWfreq <- as.matrix(clean_train98)

#combine the sentiment with the term frequencies of the BoW in training data into a dataframe
sentiment<- train_data$sentiment
train_data98_m <- data.frame(sentiment, train98_BoWfreq)
summary(train_data98_m)

#We will save the Bag of Words generated from the training set for later use
train98_BoWfreq_m <- findFreqTerms(train.tfidf.98)
length(train98_BoWfreq_m)

###(2) Processing test data: test_data_m

#To generate test_data_m that include test's term frequency of words selected from the train data
test_data <- read.csv("test.csv", header=T, stringsAsFactors = FALSE)
str(test_data)

#first create a corpus of review document in test
test_corpus_m <- Corpus(DataframeSource(as.matrix(test_data$text)))

#generate test1's Document Term Matrix based on the BoW decided by the training data
BoW_test_m <-DocumentTermMatrix(test_corpus_m, control = list(tolower = T,
             removeNumbers=T, removePunctuation=T, stopwords=T, stripWhitespace=T,
             stemming=T, dictionary = train98_BoWfreq_m))
str(BoW_test_m)
dim(BoW_test_m) 

#transform the format from Document Term Matrix to Matrix
test_BoWfreq_m <- as.matrix(BoW_test_m)

#combine the sentiment with the term frequencies  of the BoW in test
#into a data frame
test_data_m <- data.frame(test_BoWfreq_m)
str(test_data_m)
summary(test_data_m)

###(3) Fitting models and prediction:

## Model 1. Multinomial Logistic Regression Model
library("nnet")
train_data98_m$sentiment <- as.factor(train_data98_m$sentiment)
model.lg <- multinom(sentiment ~ ., data = train_data98_m)
summary(model.lg)

#evaluation of prediction accuracy
train_Pred <- predict(model.lg, newdata = train_data98_m)
t <- table(train_Pred, train_data$sentiment)
#train_Pred   1   2   3   4   5
#      1      2   1   1   0   0
#      2      0   7   6   2   0
#      3      19 104 582 153  48
#      4      2   4  13  22   4
#      5      0   1   1   2   7

(2+7+582+22+7)/981  #[1] 0.6320082

# LOOCV
positive <- 0

for (idx in 1:nrow(train_data98_m)){
  
  train <- train_data98_m[-idx,]
  test <- train_data98_m[idx,]
  
  model.lg <- multinom(sentiment ~ ., data = train)
  
  pred<-predict(model.lg, newdata = test) 
  #table<- table(pred,train_data98_m[idx,]$sentiment)
  
  if (pred == train_data98_m[idx,]$sentiment){
  positive <- positive + 1
  }
}
  
accuracy <- positive/nrow(train_data98_m)
accuracy

#LOOCV
cvv <- sapply(X = 1:nrow(train_data98_m), function(X){
  training_data <- train_data98_m[-X,]
  testing_data <- train_data98_m[X,]
  cvmodel <- multinom(sentiment ~ ., data = training_data)
  guess <- predict(cvmodel, newdata=testing_data)
  accuracy <- guess == testing_data$sentiment
  return(accuracy)
})

totalaccuracy <- sum(cvv) / length(cvv)
#[1] 0.5891947

#prediction of test sentiment
test_Pred <- predict(model.lg, newdata = test_data_m)

#export into csv file
id<- test_data$id
sentiment<- test_Pred
test.score <- cbind(id, sentiment)
test.score<-as.data.frame(test.score)
write.csv(test.score,"yh8a_twitter_logistic_submission3i.csv")


# Model 2. Linear Regression Model
train_data98_m$sentiment<-as.numeric(train_data98_m$sentiment)
train.lm <- glm(sentiment~., data=train_data98_m)
summary(train.lm)

#LOOCV
cv.err =cv.glm(train_data98_m, train.lm)
cv.err$delta
#[1] 0.6212805 0.6212479

#percentage of correct prediction in training data
train.pred <- round(predict(train.lm, newdata=train_data98_m))
table(train.pred, train_data$sentiment)

mean(train.pred== train_data$sentiment)
#accuracy 0.6167176

#Prediction of test data:
test_Pred2 <- abs(round(predict(train.lm, newdata = test_data_m)))
test_Pred2
 
#Export to csv file
id<- test_data$id
sentiment<- as.data.frame(test_Pred2)
test.score2 <- cbind(id, sentiment)
test.score2<-as.data.frame(test.score2)
write.csv(test.score2,"yh8a_twitter_Linear_submission2i.csv")

