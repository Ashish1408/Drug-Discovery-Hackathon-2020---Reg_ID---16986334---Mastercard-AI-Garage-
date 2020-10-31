import pandas as pd
import numpy as np

#Read Testing Data and Prepare SMI Files

df_data = pd.read_csv("../External Test Set/External_Test_Set.csv")

smi = np.array(df_data["Canonical SMILES"]).astype("str")

print("Sample Raw Data - \n",df_data.head())

cid = df_data["Canonical SMILES"]

c = 0
for i in cid:
    with open('../SMI Files/'+str(i)+'.smi', 'w') as outfile:
        outfile.write(smi[c])
    c = c + 1

print("SMI Files Created")
