import pandas as pd
import numpy as np

#Preparing Input Data

df_pdl_data = pd.read_csv("../Padel Output/Padel_Output_Data.csv")
df_raw_data = pd.read_csv("../External Test Set/External_Test_Set.csv")

df_pdl_data["CID"] = df_pdl_data['Name'].apply(lambda x: str(x[8:]))
df_pdl_data = df_pdl_data.drop(["Name"],axis = 1)

df_raw_data["CID"] = df_raw_data["Canonical SMILES"]

df_input_data = pd.merge(df_pdl_data, df_raw_data,  how='inner', left_on=["CID"], right_on = ["CID"])

df_input_data = df_input_data.drop(["CID"],axis = 1)
print("Input shape - \n",df_input_data.shape)
print("Input Data - \n",df_input_data.head())

print("Input Data Prpared")
df_input_data.to_csv("../Input Data/Input_Data.csv")
print("Input Data Saved")