# Adrian Mead, Myron Chang, Yi Hao
# SYS class
# Kaggle Competition 3 - Twitter Self-driving Car Sentiment
# September 26, 2017

# THESE ARE THE MAJOR LEVERS AVAILABLE TO US
# Play around with weighting schemes
# Play around with sparcity filter
# Play around with k-values
# Consider constructing bigrams
# Distance measurement -- also angle (cosine similarity)

# Load in the libraries
library(tidyverse)
library(tm)

# Read in the training/testing data
train_data <- read_csv(file = 'train.csv')
test_data <- read_csv(file = 'test.csv')

# When dealing with text we only need the character column
train_tweets <- as.tibble(train_data$text)
test_tweets <- as.tibble(test_data$text)

# convert this part of the data frame to a corpus object.
twitter_corpus <- VCorpus(DataframeSource(train_tweets))
test_corpus <- VCorpus(DataframeSource(test_tweets))

# Playing around with weightings
#lapply(X = c('weightTfIdf', 'weightTf', 'weightBin', 'weightSMART'), function(Z){
# I tried all 4 possible weightings to see which one yielded the best results. weightTfIdf was the best
  # compute TF-IDF matrix and inspect sparsity
  # twitter.tfidf = DocumentTermMatrix(twitter_corpus, control = list(weighting = get(Z)))
  twitter.tfidf = DocumentTermMatrix(twitter_corpus, control = list(weighting = weightTfIdf))
  test.tfidf = DocumentTermMatrix(test_corpus, control = list(weighting = weightTfIdf))
  twitter.tfidf  # non-/sparse entries indicates how many of the DTM cells are non-zero and zero, respectively.
  test.tfidf
  # sparsity is number of non-zero cells divided by number of zero cells.
  
  ##### Reducing Term Sparsity #####
  
  # there's a lot in the documents that we don't care about. clean up the corpus.
  twitter_corpus.clean = tm_map(twitter_corpus, stripWhitespace)                          # remove extra whitespace
  twitter_corpus.clean = tm_map(twitter_corpus.clean, removeNumbers)                      # remove numbers
  twitter_corpus.clean = tm_map(twitter_corpus.clean, removePunctuation)                  # remove punctuation
  twitter_corpus.clean = tm_map(twitter_corpus.clean, content_transformer(tolower))       # ignore case
  twitter_corpus.clean = tm_map(twitter_corpus.clean, removeWords, stopwords("english"))  # remove stop words
  twitter_corpus.clean = tm_map(twitter_corpus.clean, stemDocument)                       # stem all words
  # Now manipulate the test data the same way
  test_corpus.clean = tm_map(test_corpus, stripWhitespace)                          # remove extra whitespace
  test_corpus.clean = tm_map(test_corpus.clean, removeNumbers)                      # remove numbers
  test_corpus.clean = tm_map(test_corpus.clean, removePunctuation)                  # remove punctuation
  test_corpus.clean = tm_map(test_corpus.clean, content_transformer(tolower))       # ignore case
  test_corpus.clean = tm_map(test_corpus.clean, removeWords, stopwords("english"))  # remove stop words
  test_corpus.clean = tm_map(test_corpus.clean, stemDocument)                       # stem all words
  
  twitter_corpus[[1]]$content
  twitter_corpus.clean[[1]]$content  # do we care about misspellings resulting from stemming?
  test_corpus[[1]]$content
  test_corpus.clean[[1]]$content  # do we care about misspellings resulting from stemming?
  
  # recompute TF-IDF matrix
  # twitter.clean.tfidf = DocumentTermMatrix(twitter_corpus.clean, control = list(weighting = get(Z)))
  twitter.clean.tfidf = DocumentTermMatrix(twitter_corpus.clean, control = list(weighting = weightTfIdf))
  test.clean.tfidf = DocumentTermMatrix(test_corpus.clean, control = list(weighting = weightTfIdf))
  
  # reinspect the first 5 documents and first 5 terms
  twitter.clean.tfidf[1:5,1:5]
  as.matrix(twitter.clean.tfidf[1:5,1:5])
  test.clean.tfidf[1:5,1:5]
  as.matrix(test.clean.tfidf[1:5,1:5])
  
  # we've still got a very sparse document-term matrix. remove sparse terms at various thresholds.
  # I went through to find the sensitivity to varying sparsity
  # most_of_the_way <- bind_rows(lapply(X = seq(.9, .995, .005), function(X){
  # I was curious to see what sparsity would produce the best test accuracy. It turns out the closer to 1 you got (more terms), the better
    # twitter.tfidf.sparse = twitter.clean.tfidf
    # twitter.tfidf.sparse = removeSparseTerms(twitter.clean.tfidf, X)  # remove terms that are absent from at least X% of documents (keep most terms)
    twitter.tfidf.sparse = removeSparseTerms(twitter.clean.tfidf, .995)  # remove terms that are absent from at least 99.5% of documents (keep most terms)
    twitter.tfidf.sparse
    # DON'T REMOVE SPARSE TERMS FROM THE TEST DATA! IT ISN'T NECESSARY AS WE'RE NOT BUILDING A MODEL FROM THOSE TERMS
    # as.matrix(twitter.tfidf.sparse[1:5,1:5])

    # Let's try out the function with some data. This let me test my function with a few easy sanity-check inputs
    # train <- twitter.tfidf.sparse
    # test <- test.clean.tfidf
    # prediction_column <- train_data$sentiment
    # k <- 1
    
    # We have a vectorised mode function for quickly calculating the mode
    # Taken from tutorialspoint at
    # https://www.tutorialspoint.com/r/r_mean_median_mode.htm
    getmode <- function(nums) {
      uniqnums <- unique(nums)
      uniqnums[which.max(tabulate(match(nums, uniqnums)))]
    }
    
    # Also interested in Cosine Distance. This implementation was found on stackoverflow at 
    # https://stackoverflow.com/questions/2535234/find-cosine-similarity-between-two-arrays
    cosineDist <- function(x){
      as.dist(1 - x%*%t(x)/(sqrt(rowSums(x^2) %*% t(rowSums(x^2))))) 
    }
    
    #Now going to code up KNN -- includes LOOCV insideit when test = NA (so when you don't pass a testing set to it)
    funcKNN <- function(train, test = NA, prediction_column, k){
      mtrx.tfidf = as.matrix(train)
      # dist.matrix = as.matrix(dist(mtrx.tfidf))
      dist.matrix = as.matrix(cosineDist(mtrx.tfidf)) # Cosine distance ended up producing slightly better accuracy on the test data
      # Some preliminary work follows on the next two lines for trying to reweight counts (this was not successful at all)
      # grpd_sentiment <- group_by(train_data, sentiment)
      # sentiment_percent <- summarise(grpd_sentiment, percent = n() / NROW(grpd_sentiment))
      # Perform cross-validation on the full set of data when test is NA
      if(is.na(test)){
        # Default to LOOCV
        # I also tried cosine distance as a distance metric; it was not particularly useful
        # LOOCV is pretty easy; we don't need to do anything random as the process is deterministic (so no sample function req)
        correct_pred <- sapply(X = 1 : NROW(dist.matrix), function(X){
          # Alright, so the general order here is that the function will go through each row of the distance matrix one at a time
          # and use that row as the test set while all of the other rows are used as the training set
          # Distance matrix
          most.similar.documents = order(dist.matrix[X,], decreasing = FALSE)
          # Use k to find the k-nearest neighbor -- ignore the first one (the test row is closest to itself)
          knn <- most.similar.documents[2:(k+1)]
          # Now pick the value with the most counts
          
          # I did attempt a normalization technique to re-weight the counts of the different sentiments
          # prediction_values <- prediction_column[knn]
          # prediction_value <- which.max(tabulate(match(prediction_values, sentiment_percent$sentiment)) / sentiment_percent$percent)
          
          prediction_value <- getmode(prediction_column[knn])
          # This has you pick the mean of the values (another option itself)
          # prediction_value <- round(mean(prediction_column[knn])) # Old method before when we weren't taking weighting into account
          # return(prediction_value)
          return(prediction_value == prediction_column[X])
        })
        accuracy <- sum(correct_pred) / length(correct_pred)
        accuracy
        # hist(correct_pred) # I wanted to track the distribution of predictions to make sure it wasn't just all 3's
        return(accuracy)
      }
      # Now dealing with the case where we actually want to run the model on some testing set
      else{
        mtrx.test <- as.matrix(test)
        # The next few lines make sure that we only use features from the training set that are also present in the 
        # testing set. o/w you get an error 
        train_column_names <- colnames(mtrx.tfidf)
        test_column_names <- colnames(mtrx.test)
        shared_column_names <- test_column_names[test_column_names %in% train_column_names]
        # Now perform KNN
        sapply(X = 1 : NROW(mtrx.test), function(X){
          # The idea here is that we go through each row of the testing set one at a time and find the distances between it and 
          # every row in the training set. Then we can make a prediction for that test data based on its nearest neighbors.
          train_and_test <- rbind(mtrx.test[X,shared_column_names], mtrx.tfidf[,shared_column_names])
          # Distance matrix
          dist.matrix = as.matrix(cosineDist(train_and_test))
          most.similar.documents = order(dist.matrix[1,], decreasing = FALSE)
          # Use k to find the k-nearest neighbor -- ignore the first one (the test row is closest to itself)
          knn <- most.similar.documents[2:(k+1)]
          # Now pick the value with the most counts
          prediction_value <- getmode(prediction_column[knn])
          return(prediction_value)
        })
       }
    }
    # k_outputs <- bind_rows(lapply(X = 1:50, function(Y){ # The goal here was to see how sensitive the test accuracy was to varying values of k
    #   pred_accuracy <- funcKNN(twitter.tfidf.sparse, prediction_column = train_data$sentiment, k = Y)
    #   return(tibble(acc = pred_accuracy, k = Y))
    # }))
    # k_outputs$scarcity <- X
    # return(k_outputs)
  # }))
  # most_of_the_way$weighting <- Z
  # return(most_of_the_way)
# })
# output_kays <-sapply(X = ((1:50) * 3) - 2, function(X){funcKNN(twitter.tfidf.sparse, NA, train_data$sentiment, X)})
# plot(output_kays) # I wanted to visualize how k actually affected the predicted test accuracy

# I ran two versions of funcKNN. One with Euclidean distance and the other with Cosine distance. I wanted to see which produced 
# the better prediction accuracy during CV
# cosine_dist# This is with cosinedist
# cosine_dist$dist <- 'cosine'
# WeightTfIdf is clearly the best
# euclid_dist
# euclid_dist$dist <- 'euclidean'

# survey_of_fits <- bind_rows(cosine_dist, euclid_dist)

# I plotted all of my different CV outcomes when adjusting all of the obvious levers I mentioned at the very beginning
# The graph is faceted by value of k horizontally and by the distance metric vertically
# The x-axis has the scarcity cutoff
# The y-axis has the predicted accuracy based on LOOCV
# And the color is by weighting used in the term doc matrix
# library(ggplot2)
# ggplot(survey_of_fits, aes(x = scarcity, y = acc, color = weighting)) + 
#   facet_grid(dist~k) + 
#   scale_y_continuous("Accuracy") + 
#   scale_x_continuous("Scarcity cutoff") +
#   geom_point(alpha = .75) +
#   theme_bw()
# The results were not particularly encouraging. You can see immediately that you get a big performance boost from 
# using weightTfIdf as opposed to any other weighting. You also see that the higher k-values produce the larger 
# accuracies when scarcity cutoff is closest to 1. The distance metrics don't seem to be particularly better than one another. 
# However, following the survey of all of all of these cross-validated accuracies, I found that the best accuracy came 
# from k = 9, dist = 'cosine', scarcity = .995, and weighting = 'weightTfIdf'
    
# Running KNN on the test data using the training data -- takes a while to run
test.predictions <- funcKNN(twitter.tfidf.sparse, test.clean.tfidf, train_data$sentiment, k = 9)

# Final formatting for submission
submit_this <- test_data
submit_this$sentiment <- test.predictions
submit_this$text <- NULL
submit_this
write_csv(submit_this, 'nonparam_knn.csv')
