PROBLEM STATEMENT
A majority of customers rely on the review of the product on the websites which helps in forming an opinion about the product. Thus, positive or negative reviews have direct influence on the product and this makes online reviews an integral part of the business. This, unfortunately also gives strong incentives for opinion spamming and thus detection of online review spam becomes important. Such individuals are called opinion spammers and their activities are called opinion spamming. Our goal is to devise a technique to detect with certain degree of certainty which reviews are spam, so that the online review website can take proper action. 
DATA SET DESCRIPTION
The dataset consists of reviews on hotels.
The dataset is imported and stored in three columns: 
●	Polarity of the review
●	Review itself
●	True or Deceptive as ('t' or 'd')
DESCRIPTION OF FEATURES SELECTED AND PRE-PROCESSING
The true value 't' is converted to 1 and deceptive value 'd' to 0 because they will be used as target value and the review as feature.
Then the Review data is split into testing data and training data (0.3 and 0.7 respectively).
Then the CountVectorizer() function is used to extract numeric features of each of the review as classifier can only use numeric data to compute something.
CountVectorizer():
Converts a collection of text documents to a matrix of token counts
ALGORITHM USED
Multinomial Naïve Bayes algorithm is used as a classifier to classify the reviews as Deceptive/True.

