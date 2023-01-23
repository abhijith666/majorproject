import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.svm import SVR

def importdata():
	balance_data = pd.read_csv(
'eGeMAPSv02_adcn_01.csv',
	sep= ',', header = None)
	
	# Printing the dataswet shape
	print ("Dataset Length: ", len(balance_data))
	print ("Dataset Shape: ", balance_data.shape)
	
	# Printing the dataset obseravtions
	print ("Dataset: ",balance_data.head())
	return balance_data

def splitdataset(balance_data):

	# Separating the target variable
	X = balance_data.values[:, 0:87]
	Y = balance_data.values[:, 88]
    

	# Splitting the dataset into train and test
	X_train, X_test, y_train, y_test = train_test_split(
	X, Y, test_size = 0.3, random_state = 100)
	
	return X, Y, X_train, X_test, y_train, y_test


def train_using_rbf(X_train, X_test, y_train):

	# Creating the classifier object
	clf_rbf = SVC(kernel = 'rbf', random_state = 50)

	# Performing training
	clf_rbf.fit(X_train, y_train)
	return clf_rbf

# def train_using_svr(X_train, X_test, y_train):

# 	# Creating the classifier object
# 	clf_svr = SVR(kernel='rbf')

# 	# Performing training
# 	clf_svr.fit(X_train, y_train)
# 	return clf_svr



 
def prediction(X_test, clf_object):

	# Predicton on test with giniIndex
	y_pred = clf_object.predict(X_test)
	print("Predicted values:")
	print(y_pred)
	return y_pred
	
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
	
	print("Confusion Matrix: ",
		confusion_matrix(y_test, y_pred))
	
	print ("Accuracy : ",
	accuracy_score(y_test,y_pred)*100)
	
	print("Report : ",
	classification_report(y_test, y_pred))

# Driver code
def main():
	
	# Building Phase
	data = importdata()
	X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
	clf_rbf = train_using_rbf(X_train, X_test, y_train)
	# clf_svr = train_using_svr(X_train, X_test, y_train)
	
	# Operational Phase
	print("Results Using rbf Index:")
	
	# Prediction using gini
	y_pred_rbf = prediction(X_test, clf_rbf)
	cal_accuracy(y_test, y_pred_rbf)
	
	# print("Results Using svr:")
	# # Prediction using entropy
	# y_pred_svr = prediction(X_test, clf_svr)
	# cal_accuracy(y_test, y_pred_svr)
	
	
	




	
	
	
	
# Calling main function
if __name__=="__main__":
	main()
