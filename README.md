# sys6018-competition-twitter-sentiment
Competition 3:  Predicting Twitter Sentiment (Competition 3-7)


Teammate roles:
Yi Hao - Data cleaning, multinomial logistic regression (including respective cross-validation)
Myron Chang - GitHub coordinator, lead parametric linear model approach (including respective cross-validation)
Adrian Mead - Lead nonparametric KNN approach (including respective cross-validation)


Scores using nonparametric KNN and parametric linear models:

Parametric:
Public score: 0.68507
Private score: 0.65510

Nonparametric:
Public score: 0.65848
Private score: 0.64693


Reflection questions:
#1. Who might care about this problem and why?
#2. Why might this problem be challenging?
#3. What other problems resemble this problem?


1.	Automobile company executives may care about this particular problem because they may consider whether to make a strategic decision to make and sell driverless cars based on the results from this problem. If Twitter sentiment is mostly positive, the executives may infer that there is a market for driverless cars and decide to get in that market.  On the other hand, if Twitter sentiment is neutral or negative, then they may decide not to pursue this market. With an accurate predictive model from this Kaggle competition, the executives can keep an eye on current, recent tweets and essentially know the current sentiment of society on driverless cars. Lawmakers and legislators may also care about this problem for policy reasons; they will push for laws corresponding to how they perceive their constituents feel about driverless cars. Additionally, researchers who specialize in text analysis may care about this problem, not specifically with regards to driverless cars but rather how this problem and the model generated from it can be tailored to other text mining problems.
2.	As we saw, sentiment analysis for text is a particularly tricky problem. It’s an issue that manages to combine high degrees of dimensionality with a great number of rules and semantic relationships to be kept track of. Order matters, emphasis matters, capitalization matters; the number of variables to pay attention to is staggering. At the same time it’s also fairly subjective; it might not be too difficult to categorize a tweet as positive or negative, but having one person say that a sentiment is a 4 as opposed to a 5 is a bit more nuanced and difficult and is subject to the biases and training processes that go into having humans produce ground truth labels. Not only that, but the relationships between text and score appear to be difficult to frame parametrically, so we likely need to use nonparametric models which typically require a large amount of training data to achieve reasonable accuracy, which itself requires a budget allocation.
3.	Probably the problem which most closely resembles sentiment analysis for text would be picture categorization. That is, feeding a raw image into a model and having it correctly identify what object is in the image. The same sort of difficulties exist here as in text mining; you have a large amount of dimensionality to your data (in the case of images you could have every pixel be a variable). There’s also a great deal of semantic processing that our eyes/brains do for us, such as identifying edges in an image, color gradients, varying shapes and sizes, which come together to form an image. Likewise, there is no clear parametric solution here; indeed people have turned to increasingly flexible and uninterpretable models like convolutional neural networks (CNN), to try and produce more accurate results.
