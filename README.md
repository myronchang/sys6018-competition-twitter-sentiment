# sys6018-competition-twitter-sentiment
Competition 3:  Predicting Twitter Sentiment (Competition 3-7)


Teammate roles:
Yi Hao - Data cleaning
Myron Chang - GitHub coordinator, lead parametric linear model approach (including respective cross-validation)
Adrian Mead - Lead nonparametric KNN approach (including respective cross-validation)




Grading

Everyone on your team will receive the same grade, which will be assessed on the basis of evidence found for the following key activities:
•Data exploration
•Data cleaning (missing values, outliers, etc.)
•Rationale for the selected statistical modeling methods
•Correct implementation and use of statistical modeling methods
•Appropriate model selection approach (train/test, cross-validation, etc.)
•Thoroughly documented code (R comments)
•Appropriate use of functions to reduce code complexity and redundancy
•Clear thinking about the reflection questions

•A 1-page reflection on the following questions.
◦Who might care about this problem and why?
◦Why might this problem be challenging?
◦What other problems resemble this problem?





# Parametric models having difficulty because they are biased wrong way
# Assumes linear relationship, but probably not a linear relationship here
# Not much improvement beyond guessing mostly 3s.
# 
# Sample size is small, almost 1000 points with a high-dimensional data
# Would prefer 100000 or millions of points.







# Playing around with weightings
#lapply(X = c('weightTfIdf', 'weightTf', 'weightBin', 'weightSMART'), function(Z){
# I tried all 4 possible weightings to see which one yielded the best results. weightTfIdf was the best




  # I went through to find the sensitivity to varying sparsity
  # most_of_the_way <- bind_rows(lapply(X = seq(.9, .995, .005), function(X){
  # I was curious to see what sparsity would produce the best test accuracy. It turns out the closer to 1 you got (more terms), the better




