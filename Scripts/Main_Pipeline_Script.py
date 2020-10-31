import pandas as pd
import numpy as np
from numpy.random import seed

import os.path as op
import os

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

import tensorflow as tf

import keras
from keras import optimizers
from keras.models import load_model
from keras.models import model_from_json
from keras.models import Sequential
from keras.models import Model
from keras.layers import Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers import Input, Dense
from keras.layers import concatenate
from keras.callbacks import EarlyStopping

from catboost import Pool, CatBoostClassifier, cv
import pickle
import sys

file_path = "../Models/Catboost Classifier Model/Catboost PDL(8) + UGRNN(5) Features.pkl"

smiles = pd.read_csv("../Input Data/Input_Data.csv",index_col = ['Unnamed: 0'])
print(smiles.head())

fp_type = 4 #Other - Using PaDel Features + Graph Kernel UGRNN Features

def load_model(file):
	#load model from pickle file
	if (op.exists(file) and op.isfile(file)):
		model = joblib.load(file)
		print("Loaded Model from disk")
	else:
		print("File does not exist or is corrupted")
		return()
	return(model)

def create_descriptor(smiles,choice):
	# Load Encoder model from the file
	json_file = open('../Models/Auto-Encoder Model/Autoencodder Model + 1444 Padel Features.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	encoder = model_from_json(loaded_model_json)
	# load weights into new model
	encoder.load_weights("../Models/Auto-Encoder Model/Autoencodder Model + 1444 Padel Features.h5")
	print("Loaded Auto-encoder model from disk")

	# Load Standard Scaler Model
	scaler = pickle.load(open('../Models/Standard Scaler Model/Standard_Scaler.pkl','rb'))
	print("Loaded Standard Scaler model from disk")

	data = smiles.drop(["Canonical SMILES","Label"],axis = 1)
	data.shape


	data_scaled = scaler.transform(data)

	encoded_data = pd.DataFrame(encoder.predict(data_scaled))
	encoded_data = encoded_data.add_prefix('feature_')

	encoded_data["Canonical SMILES"] = smiles["Canonical SMILES"]

	ugrnn_data = pd.read_csv("../UGRNN/ugrnn/data/DILI/Final_data/Predictions/UGRNN Encoddings.csv")
	
	final_data = pd.merge(ugrnn_data, encoded_data,  how='left', left_on=["Canonical SMILES"], right_on = ["Canonical SMILES"])

	final_data = final_data.drop(["Canonical SMILES"],axis = 1)

	if choice == 4:
		descriptor = final_data.copy()
		return(descriptor)
	else:
		print ('Invalid fingerprint choice')

def Save_Results(pred,probs):

	pred1 = pred
	probs1 = probs[:,1]   

	pred2 = 1 - pred
	probs2 = 1 - probs[:,1]

	#Saving Final Results
	df = pd.DataFrame()
	d = pd.read_csv("../UGRNN/ugrnn/data/DILI/Final_data/Predictions/UGRNN Encoddings.csv")
	df["Canonical SMILES"] = d["Canonical SMILES"]

	df = pd.merge(df, smiles[["Canonical SMILES","Label"]],  how='left', left_on=["Canonical SMILES"], right_on = ["Canonical SMILES"])

	label = df["Label"]

	roc_auc1 = roc_auc_score(label, probs1)
	roc_auc2 = roc_auc_score(label, probs2)

	if roc_auc1 > roc_auc2:
		df["Predictions"] = pred1
		df["Probability"] = probs1
		df.to_csv("../External Test Set Results/External_Test_Set_Result_Predictions.csv")
		calculate_metrics(label,pred1,probs1)
	else:
		df["Predictions"] = pred2
		df["Probability"] = probs2
		df.to_csv("../External Test Set Results/External_Test_Set_Result_Predictions.csv")
		calculate_metrics(label,pred2,probs2)

	print("External Test Set Results Saved")


def calculate_metrics(label,pred,probs):

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

	df_metrics.to_csv("../External Test Set Results/External_Test_Set_Results_Metrics.csv")
	print("External Test Set Result Metrics Saved")


if __name__ == '__main__':
	#perform classification for a given SMILES with a saved model

    model = load_model(file_path)

    #NECESSARY - RUN BELOW CODE FOR GENERATING GRAPH KERNEL UGRNN FEATURES
    #Trains UGRNN Model on our previous Train and Validation Set
    os.chdir("../UGRNN/ugrnn/")
    if (os.system('python train.py') == 0):
    	print("UGRNN Embeddings Generated Sucessfully")
    else:
    	print("UGRNN Embeddings Generation Failed")
    	sys.exit("UGRNN Embeddings Generation Failed")
    os.chdir("../../Scripts/")

    fp = create_descriptor(smiles,fp_type)
    pred = model.predict(fp)
    probs = model.predict_proba(fp)
    
    Save_Results(pred,probs)

    print("Results Saved")