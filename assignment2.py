##### HOUSEKEEPING #####
#ignore that weird warning when running LinearRegression(), nvm doesn't work
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

#Importing the essential libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##### PREPROCESSING #####

#1. Import dataset
dataset = pd.read_csv("housing.data.txt", sep="\s+", header=None)
dataset.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

#set up independent (x), what we'll use to predict & dependent variables (y), what we're trying to predict
x = dataset.iloc[:, [5,12]].values #[all the rows, only columns 6 & 3]: RM & LSTAT
y = dataset.iloc[:, [13]].values  #[all the rows, only column 13]: MEDV
#print(x)
#print(y)

#2. Missing data? no
#3. Categorical Encoding? not needed

#4. split into training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) #20% of the dataset is put into test set, 80% into training set
#REMOVE LATER: random state = 0 for now (for testing)

#5. Feature Scaling? no, the library will take care of this for us

##### MAKING THE MULTIPLE LINEAR REGRESSION MODEL #####
#The Model will be predicted MEDV = constant0 + (constant1 * RM) + (constant2 * LSTAT)
#	need to find constant0, constant1, & constant2. RM & LSTAT are the inputs we're trying to use to predict the MEDV

#1. fitting model to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) # we now have a model that fits the training set!

#2. test performance with test set
y_pred = regressor.predict(x_test)

#displaying results
#hard to plot graph since its 3 dimensions
open("output.txt", "w").close() #erase contents
print ("real MEDV:\tpredicted MEDV", file=open("output.txt", "a"))
import csv 
with open("output.txt", "w") as f:
	writer = csv.writer(f, delimiter="\t")
	writer.writerow(("Real MEDV values","predicted MEDV values"))
	writer.writerows(zip(y_test,y_pred))







