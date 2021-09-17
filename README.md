# credit-Card-Approval-Predictor-in-Python
Automated  Credit Card Approval Predictor for Commercial Banks

In this notebook, we will build an automatic credit card approval predictor using machine learning techniques, just like the real banks do. Most of this notebook has been studied from the project in Datacamp: Predicting Credit Card Approvals.  Part of the project has been done by Datacamp and the programs have been completed by the author.

We'll use the Credit Card Approval [dataset](http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/)  from the UCI Machine Learning Repository . The structure of this notebook is as follows:

First, we will start off by loading and viewing the dataset.
We will see that the dataset has a mixture of both numerical and non-numerical features, that it contains values from different ranges, plus that it contains a number of missing entries.
We will have to preprocess the dataset to ensure the machine learning model we choose can make good predictions.
After our data is in good shape, we will do some exploratory data analysis to build our intuitions.
Finally, we will build a machine learning model that can predict if an individual's application for a credit card will be accepted.

**1. Load and view data set. View code [here](coding/load-data)**

We find there are fifteen columns in the first five lines: 
![image](https://user-images.githubusercontent.com/53232113/133694972-c57c555c-aea7-4f52-b841-da9ec9c81549.png)


And the structure is as follows: 

Number of Instances: 690

A.  Number of Attributes: 15 + class attribute

B.  Attribute Information:

    A1:	b, a.
    A2:	continuous.
    A3:	continuous.
    A4:	u, y, l, t.
    A5:	g, p, gg.
    A6:	c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
    A7:	v, h, bb, j, n, z, dd, ff, o.
    A8:	continuous.
    A9:	t, f.
    A10:	t, f.
    A11:	continuous.
    A12:	t, f.
    A13:	g, p, s.
    A14:	continuous.
    A15:	continuous.
    A16: +,-         (class attribute)

C.  Missing Attribute Values:
    37 cases (5%) have one or more missing values.  The missing
    values from particular attributes are:

    A1:  12
    A2:  12
    A4:   6
    A5:   6
    A6:   9
    A7:   9
    A14: 13

D.  Class Distribution
  
    +: 307 (44.5%)
    -: 383 (55.5%)

**2. Inspecting the applications**
The output may appear a bit confusing at its first sight, but let's try to figure out the most important features of a credit card application.  The probable features in a typical credit card application are Gender, Age, Debt, Married, BankCustomer, EducationLevel, Ethnicity, YearsEmployed, PriorDefault, Employed, CreditScore, DriversLicense, Citizen, ZipCode, Income and finally the ApprovalStatus. 
As we can see from our first glance at the data, the dataset has a mixture of numerical and non-numerical features. 

To inspect the data we use pandas commands: describe, info and tail View code [here](coding/inspect)

The output is:
For cc_apps.describe():

![image](https://user-images.githubusercontent.com/53232113/133695656-f0d0975f-0ca9-493d-babb-a0adc93de0b6.png)


The features 2, 7, 10 and 14 contain numeric values (of types float64, float64, int64 and int64 respectively) and all the other features contain non-numeric values. Apart from these, we can get useful statistical information (like mean, max, and min) about the features that have numerical values.

For cc_apps.info():

![image](https://user-images.githubusercontent.com/53232113/133699014-cc0b4e32-d622-41f7-9e8a-293bd45ce1e9.png)

As seen, features 2(debt) and 7(years employed) have type "float64" and features 10(credit score) and 14(income) have type "int64". These onlyh four features are numerical. The remaining features are type "object", which are non-numeric values.  Just observe that the feature 15('approval status') is the target which is a non-numeric value.

In order to check the last 17 lines of data in dataframe we use pandas command "cc_apps.tail(17)". The resuls is as follows:
							
![image](https://user-images.githubusercontent.com/53232113/133701389-15624bf1-de3d-484c-93e9-45a68566b982.png)

The dataset also contains values from several ranges. Some features have a value range of 0 - 28, some have a range of 2 - 67, and some have a range of 1017 - 100000. 

**3 Handling missing values**
**A. Changing '?' with 'NaN'**
The dataset has missing values, which we'll take care of in this task. The missing values in the dataset are labeled with '?', which can be seen in raw 673, column 0 which refers to "gender" feature:

![image](https://user-images.githubusercontent.com/53232113/133701790-dca21919-053a-4b76-ba0d-534cce434856.png)


Now, let's temporarily replace these missing value question marks with NaN. View code [here](coding/missing1).

Now, check how raw 673, column 0("Gender") has changed to "NaN'

![image](https://user-images.githubusercontent.com/53232113/133703535-76fe61aa-3a8e-4aaf-87a1-832c02288e82.png)


