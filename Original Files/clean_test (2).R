library(XML)
library(tm)
library(tidyverse)


test.data <- read.csv("test.csv", stringsAsFactors = FALSE) 
test.df<-as.data.frame(test.data$text, stringsAsFactors = FALSE)

news = VCorpus(DataframeSource(test.df))

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
cleanedtest98 <- as.data.frame(as.matrix(tfidf.98))

