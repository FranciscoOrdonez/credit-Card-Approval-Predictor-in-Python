# credit-Card-Approval-Predictor-in-Python
Automated  Credit Card Approval Predictor for Commercial Banks

In this notebook, we will build an automatic credit card approval predictor using machine learning techniques, just like the real banks do. Most of this notebook has been studied from the project in Datacamp "Predicting Credit Card Approvals".  Part of the project has been done by Datacamp and the programs have been completed by the author.

We'll use the Credit Card Approval [dataset](http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/)  from the UCI Machine Learning Repository . The structure of this notebook is as follows:

First, we will start off by loading and viewing the data set. 
We will see that the dataset has a mixture of both numerical and non-umerical features, that it contains values from different ranges, plus that it contains a number of missing entries.
We will have to preprocess the dataset to ensure the machine learning model we choose can make good predictions.
After our data is in good shape, we will do some exploratory data analysis to build our intuitions.
Finally, we will build a machine learning model that can predict if an individual's application for a credit card will be accepted.

**1. Load and view data set**

First we have to go to UCI Machine Leaning Repository on Credit Approval Data, download Data Folder,  get 'crx.data',  save it with name 'crx.csv', and load and view the data with pandas commands. View code [here](coding/load-data)

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

As seen, features 2(debt) and 7(years employed) have type "float64" and features 10(credit score) and 14(income) have type "int64". These only four features are numerical. The remaining features are type "object", which are non-numeric values.  Just observe that the feature 15('approval status') is the target which is a non-numeric value.

In order to check the last 17 lines of data in dataframe we use pandas command "cc_apps.tail(17)". The resuls is as follows:
							
![image](https://user-images.githubusercontent.com/53232113/133701389-15624bf1-de3d-484c-93e9-45a68566b982.png)

The dataset also contains values from several ranges. Some features have a value range of 0 - 28, some have a range of 2 - 67, and some have a range of 1017 - 100000. 

**3 Handling missing values**
Missing values are important to check before applying a model,  because most models do not handle missing values implicitly.

**3-A. Changing '?' with 'NaN'**

The dataset has missing values, which we'll take care of in this task. The missing values in the dataset are labeled with '?', which can be seen in line 673, column 0 which refers to "gender" feature:

![image](https://user-images.githubusercontent.com/53232113/133701790-dca21919-053a-4b76-ba0d-534cce434856.png)


Now, let's temporarily replace these missing value question marks with NaN. View code [here](coding/missing1).

Now, check how row 673, column 0("Gender") has changed to "NaN'

![image](https://user-images.githubusercontent.com/53232113/133703535-76fe61aa-3a8e-4aaf-87a1-832c02288e82.png)

**3-B. Handling missing values in numeric columns**
First we are going to check how many missing values we have per column or feature, then, we are going to impute the missing values with a strategy called mean imputation, which only works for numeric columns, finally we will check again how many missing values we have per column. View code [here](coding/missing2)
As the dataset contains both numeric and non-numeric data, for this task we will only impute the missing values (NaNs) present in the columns having numeric data-types (columns 2, 7, 10 and 14).

The result before and after the mean imputation is:

![image](https://user-images.githubusercontent.com/53232113/133710796-49925ae6-1f52-47e5-ad2f-2eb6b8c90d0f.png)

The number of missing values per column is the same before and after the imputation, which means there has not been any missing values on numeric columns.
The missing values are on columns 0,1,3,4,5,6 and 13, which are non-numerical, with a total 67 missing values.

**3-C. Handling missing values in non-numeric columns**

There are still some missing values to be imputed for columns 0, 1, 3, 4, 5, 6 and 13. All of these columns contain non-numeric data and this is why the mean imputation strategy would not work here. This needs a different treatment.

We are going to impute these missing values with the most frequent values as present in the respective columns. This is good practice when it comes to imputing missing values for categorical data in general. to view code, go to [code](coding/missing3),

After imputing the missing values with the most frequent values in the respective columns,  the results of number of  missing values per column is as follows:

![image](https://user-images.githubusercontent.com/53232113/133713791-abbba1ba-bd18-4e72-b00a-a91d380f5c5a.png)

The result is  that there is no missing values in any columns any more. We have achived to have data with no missing values.

**4 Preprocesing the data**

For preprocessing we are going to conver non-numeric data into numeric, split the data into train and test sets, and scale the feature values to a uniform range.

**4-A  Convert non-numeric data into numeric**

First, we will be converting all the non-numeric values into numeric ones. We do this because not only it results in a faster computation but also many machine learning models (like XGBoost) (and especially the ones developed using scikit-learn) require the data to be in a strictly numeric format. We will do this by using a technique called label encoding.

To view the code for  label encoding and the statistics results, check [here](coding/preprocessing1).

If label encoding is successful, all columns will have numeric data.  To check this we print dataframe info, summary statistics and the last 17 rows of dataframe.
- Dataframe info:

![image](https://user-images.githubusercontent.com/53232113/133717485-c105d21a-d397-4ef2-b4eb-406a8b82b3de.png)

As we can see, all columns non-numeric have been changed to numeric  int32.

- summary statistics:

![image](https://user-images.githubusercontent.com/53232113/133717895-8b57c842-6890-4841-bc9e-ababaad0993f.png)


- last 17 rows: 

![image](https://user-images.githubusercontent.com/53232113/133718074-d61366c2-07fe-4e13-9bac-de840f012d4d.png)




