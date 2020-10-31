import pandas as pd
import numpy as np
from numpy.random import seed

import os
import os.path as op

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline

from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import auc,precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from imblearn.metrics import geometric_mean_score

from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score

from catboost import Pool, CatBoostClassifier, cv
import pickle

file_path = "../Models/Catboost Classifier Model/Catboost PDL(8) + UGRNN(5) Features.pkl"

def load_model(file):
	#load model from pickle file
	if (op.exists(file) and op.isfile(file)):
		model = joblib.load(file)
		print("Loaded Model from disk")
	else:
		print("File does not exist or is corrupted")
		return()
	return(model)

def Save_Results(choice,dataset,model):

	label = dataset["Label"]

	data = dataset.drop(["Canonical SMILES","Label"],axis = 1)

	#Make Predictions
	if choice == 1:
		pred = model.predict(data)
		probs = model.predict_proba(data)
		probs = probs[:,1]

	if choice == 2:
		pred = model.predict(data)
		pred = 1 - pred

		probs = model.predict_proba(data)
		probs = 1 - probs[:,1]

	df_pred = pd.DataFrame()
	df_pred["Canonical Smiles"] = dataset["Canonical SMILES"]
	df_pred["Label"] = dataset["Label"]
	df_pred["Predictions"] = pred
	df_pred["Probability"] = probs

	#Calculating Metrics
	values  = []
	accuracy = accuracy_score(label,pred)

	
	roc_auc = roc_auc_score(label, probs)

	mcc = matthews_corrcoef(label, pred)

	g_mean = geometric_mean_score(label, pred)

	
	precision = precision_score(label, pred)
	sensitivity = recall_score(label, pred)
	f1 = f1_score(label, pred)
	tn, fp, fn, tp = metrics.confusion_matrix(label, pred).ravel()
	specificity = tn / (tn+fp)

	metrics_array = ["Accuracy","ROC-AUC","F1_Score","Precision","Sensitivity","Specificity","Matthews_Correlation Coefficient ","Geometric_Mean"]

	values.append(accuracy)
	values.append(roc_auc)
	values.append(f1)
	values.append(precision)
	values.append(sensitivity)
	values.append(specificity)
	values.append(mcc)
	values.append(g_mean)

	df_metrics = pd.DataFrame()

	df_metrics["Metric"] = metrics_array
	df_metrics["Values"] = values


	if choice == 1:
		df_pred.to_csv("../Our Train and Test Results/Train_Set_Predictions.csv")
		df_metrics.to_csv("../Our Train and Test Results/Train_Set_Result_Metrics.csv")
		print("Train Set Results Saved")
	elif choice == 2:
		df_pred.to_csv("../Our Train and Test Results/Test_Set_Predictions.csv")
		df_metrics.to_csv("../Our Train and Test Results/Test_Set_Result_Metrics.csv")
		print("Test Set Results Saved")
	else:
		print("Invalid Choice")


if __name__ == '__main__':
	#perform classification for a given SMILES with a saved model

    model = load_model(file_path)

    train_set = pd.read_csv("../Our Train and Test Set/Train_Set.csv",index_col = "Unnamed: 0")

    test_set = pd.read_csv("../Our Train and Test Set/Test_Set.csv",index_col = "Unnamed: 0")

    Save_Results(1,train_set,model)
    Save_Results(2,test_set,model)
