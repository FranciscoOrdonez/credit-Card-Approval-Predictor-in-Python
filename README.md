# credit-Card-Approval-Predictor-in-Python
Automated  Credit Card Approval Predictor for Commercial Banks

In this notebook, we will build an automatic credit card approval predictor using machine learning techniques, just like the real banks do. Most of this notebook has been studied from the project in Datacamp: Predicting Credit Card Approvals.  Part of the project has been done by Datacamp and the programs have been completed by the author.

We'll use the Credit Card Approval [dataset](http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/)  from the UCI Machine Learning Repository . The structure of this notebook is as follows:

First, we will start off by loading and viewing the dataset.
We will see that the dataset has a mixture of both numerical and non-numerical features, that it contains values from different ranges, plus that it contains a number of missing entries.
We will have to preprocess the dataset to ensure the machine learning model we choose can make good predictions.
After our data is in good shape, we will do some exploratory data analysis to build our intuitions.
Finally, we will build a machine learning model that can predict if an individual's application for a credit card will be accepted.

**1. LOAD AND VIEW DATA SET. SEE CODE [HERE](coding/load-data)**

We find there are fifteen columns in the first five lines:

 0      1      2  3  4  5  6     7  8  9   10 11 12     13   14 15
 
0  b  30.83  0.000  u  g  w  v  1.25  t  t   1  f  g  00202    0  +

1  a  58.67  4.460  u  g  q  h  3.04  t  t   6  f  g  00043  560  +

2  a  24.50  0.500  u  g  q  h  1.50  t  f   0  f  g  00280  824  +

3  b  27.83  1.540  u  g  w  v  3.75  t  t   5  t  g  00100    3  +

4  b  20.17  5.625  u  g  w  v  1.71  t  f   0  f  s  00120    0  +

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
Using the output below, we can see that the outcome values in Approved are ‘+’ or ‘-’ for whether credit had been granted or not. These character symbols aren’t meaningful as is so will need to be transformed. Turning the ‘+’ to a ‘1’ and the ‘-’ to a ‘0’ will help with classification and logistic regression models later in the analysis.
data.frame':   689 obs. of  16 variables:
 $ Male          : chr  a  b ...
 $ Age           : chr  "58.67" "24.50" "27.83" "20.17" ...
 $ Debt          : num  4.46 0.5 1.54 5.62 4 ...
 $ Married       : chr  "u" "u" "u" "u" ...
 $ BankCustomer  : chr  "g" "g" "g" "g" ...
 $ EducationLevel: chr  "q" "q" "w" "w" ...
 $ Ethnicity     : chr  "h" "h" "v" "v" ...
 $ YearsEmployed : num  3.04 1.5 3.75 1.71 2.5 ...
 $ PriorDefault  : chr  f ...
 $ Employed      : chr  f z...
 $ CreditScore   : num  6 0 5 0 0 0 0 0 0 0 ...
 $ DriversLicense: chr  "f" "f" "t" "f" ...
 $ Citizen       : chr  "g" "g" "g" "s" ...
 $ ZipCode       : chr  "00043" "00280" "00100" "00120" ...
 $ Income        : num  560 824 3 0 0 ...
 $ Approved      : chr  "+" "+" "+" "+" ...

As we can see from our first glance at the data, the dataset has a mixture of numerical and non-numerical features. 

To inspect the data we use pandas commands: structure, info and tail [HERE](coding/inspect)

The output is:
For cc_apps.describe():
               2           7          10             14
count  690.000000  690.000000  690.00000     690.000000
mean     4.758725    2.223406    2.40000    1017.385507
std      4.978163    3.346513    4.86294    5210.102598
min      0.000000    0.000000    0.00000       0.000000
25%      1.000000    0.165000    0.00000       0.000000
50%      2.750000    1.000000    0.00000       5.000000
75%      7.207500    2.625000    3.00000     395.500000
max     28.000000   28.500000   67.00000  100000.000000

The features 2, 7, 10 and 14 contain numeric values (of types float64, float64, int64 and int64 respectively) and all the other features contain non-numeric values.

For cc_apps.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 690 entries, 0 to 689
Data columns (total 16 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   0       690 non-null    object 
 1   1       690 non-null    object 
 2   2       690 non-null    float64
 3   3       690 non-null    object 
 4   4       690 non-null    object 
 5   5       690 non-null    object 
 6   6       690 non-null    object 
 7   7       690 non-null    float64
 8   8       690 non-null    object 
 9   9       690 non-null    object 
 10  10      690 non-null    int64  
 11  11      690 non-null    object 
 12  12      690 non-null    object 
 13  13      690 non-null    object 
 14  14      690 non-null    int64  
 15  15      690 non-null    object 
dtypes: float64(2), int64(2), object(12)
memory usage: 86.4+ KB
None

For cc_apps.tail(17) (17 last lines of data):
    0      1       2  3  4   5   6      7  8  9   10 11 12     13   14 15
673  ?  29.50   2.000  y  p   e   h  2.000  f  f   0  f  g  00256   17  -
674  a  37.33   2.500  u  g   i   h  0.210  f  f   0  f  g  00260  246  -
675  a  41.58   1.040  u  g  aa   v  0.665  f  f   0  f  g  00240  237  -
676  a  30.58  10.665  u  g   q   h  0.085  f  t  12  t  g  00129    3  -
677  b  19.42   7.250  u  g   m   v  0.040  f  t   1  f  g  00100    1  -
678  a  17.92  10.210  u  g  ff  ff  0.000  f  f   0  f  g  00000   50  -
679  a  20.08   1.250  u  g   c   v  0.000  f  f   0  f  g  00000    0  -
680  b  19.50   0.290  u  g   k   v  0.290  f  f   0  f  g  00280  364  -
681  b  27.83   1.000  y  p   d   h  3.000  f  f   0  f  g  00176  537  -
682  b  17.08   3.290  u  g   i   v  0.335  f  f   0  t  g  00140    2  -
683  b  36.42   0.750  y  p   d   v  0.585  f  f   0  f  g  00240    3  -
684  b  40.58   3.290  u  g   m   v  3.500  f  f   0  t  s  00400    0  -
685  b  21.08  10.085  y  p   e   h  1.250  f  f   0  f  g  00260    0  -
686  a  22.67   0.750  u  g   c   v  2.000  f  t   2  t  g  00200  394  -
687  a  25.25  13.500  y  p  ff  ff  2.000  f  t   1  t  g  00200    1  -
688  b  17.92   0.205  u  g  aa   v  0.040  f  f   0  f  g  00280  750  -
689  b  35.00   3.375  u  g   c   h  8.290  f  f   0  t  g  00000    0  -


The dataset also contains values from several ranges. Some features have a value range of 0 - 28, some have a range of 2 - 67, and some have a range of 1017 - 100000. Apart from these, we can get useful statistical information (like mean, max, and min) about the features that have numerical values.

**3 Changing '?' with 'NaN'**
The dataset has missing values, which we'll take care of in this task. The missing values in the dataset are labeled with '?', which can be seen in the last cell's output:

 0      1       2  3  4   5   6      7  8  9   10 11 12     13   14 15
673  ?  29.50   2.000  y  p   e   h  2.000  f  f   0  f  g  00256   17  -

Now, let's temporarily replace these missing value question marks with NaN. See code [here](coding/missing1)

