# credit-Card-Approval-Predictor-in-Python
Automated  Credit Card Approval Predictor for Commercial Banks

In this notebook, we will build an automatic credit card approval predictor using machine learning techniques, just like the real banks do. Most of this notebook has been studied from the project in Datacamp: Predicting Credit Card Approvals.  Part of the project has been done by Datacamp and the programs have been completed by the author.

We'll use the Credit Card Approval dataset from the UCI Machine Learning Repository [data](http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/). The structure of this notebook is as follows:

First, we will start off by loading and viewing the dataset.
We will see that the dataset has a mixture of both numerical and non-numerical features, that it contains values from different ranges, plus that it contains a number of missing entries.
We will have to preprocess the dataset to ensure the machine learning model we choose can make good predictions.
After our data is in good shape, we will do some exploratory data analysis to build our intuitions.
Finally, we will build a machine learning model that can predict if an individual's application for a credit card will be accepted.

1.- load an view dataset. See code [here](coding/load-data)

We find there are fifteen columns:
 0      1      2  3  4  5  6     7  8  9   10 11 12     13   14 15
0  b  30.83  0.000  u  g  w  v  1.25  t  t   1  f  g  00202    0  +
1  a  58.67  4.460  u  g  q  h  3.04  t  t   6  f  g  00043  560  +
2  a  24.50  0.500  u  g  q  h  1.50  t  f   0  f  g  00280  824  +
3  b  27.83  1.540  u  g  w  v  3.75  t  t   5  t  g  00100    3  +
4  b  20.17  5.625  u  g  w  v  1.71  t  f   0  f  s  00120    0  +

