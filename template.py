#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def load_dataset(dataset_path):
	#To-Do: Implement this function
	#Load the csv file at the given path into the pandas DataFrame and return the DataFrame
	return pd.read_csv(dataset_path);

#def dataset_stat(dataset_df):	
	#To-Do: Implement this function
	#For the given DataFrame, return the following statistical analysis results in order
	#Number of features
	#Number of data for class 0
	#Number of data for class 1
	#return dataset_df.shape[1]-1, len(dataset_df['target']==0), len(dataset_df['target']==1)

#def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	#Splitting the given DataFrame and return train data, test data, train label, and test label in order
	#You must split the data using the given test size
	#return train_test_split(dataset_df.iloc['',''],dataset_df['target'], test_size=testset_size)

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	#Using the given train dataset, train the decision tree model
	#you can implement with default arguments
	#After the training, evaluate the performances of th emodel using the given test dataset
	#Retrun three performance metrics(accuracy, precision, recall) in order
	decisionTreeCls=DecisionTreeClassifier()
	decisionTreeCls.fit(x_train, y_train)
	predictResult=decisionTreeCls.predict(x_test)
	acc = accuracy_score(predictResult, y_test)
	prec = precision_score(predictResult, y_test)
	recall = recall_score(predictResult, y_test)
	return acc, prec, recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	randomForestCls=RandomForestClassifier()
	randomForestCls.fit=(x_train, y_train)
	predictResult=randomForestCls.predict(x_test)
	acc = accuracy_score(predictResult, y_test)
	prec = precision_score(predictResult, y_test)
	recall = recall_score(predictResult, y_test)
	return acc, prec, recall

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	pipe=make_pipeline(
		StandardScaler(),
		#RandomForestClassifier()
		SVC()
	)
	pipe.fit(x_train, y_train)
	predictResult=pipe.predict(x_test)
	acc = accuracy_score(predictResult, y_test)
	prec = precision_score(predictResult, y_test)
	recall = recall_score(predictResult, y_test)
	return acc, prec, recall

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
