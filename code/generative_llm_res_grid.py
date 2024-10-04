import numpy as np
import pandas as pd
import os
from nltk import word_tokenize

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

    
def decode(x):

    if "inclaim" in x.lower():
        return 1
    elif "outofclaim" in x.lower():
        return 0
    else: 
        return -1


acc_list = []
f1_list = []
missing_perc_list = [] 

files = os.listdir('../data/llm_prompt_output_feb_2024')

files_xls = [f for f in files if ('ec' in f and 'complex' in f and 'falcon' in f and 'few' in f)]

for file in files_xls:
    df = pd.read_csv('../data/llm_prompt_output_feb_2024/' + file)

    print(file)
    df["predicted_label"] = df["text_output"].apply(lambda x: decode(x))
    
    f1_list.append(f1_score(df["true_label"], df["predicted_label"], average='weighted'))

print("f1 score mean: ", format(np.mean(f1_list), '.4f'))
print("f1 score std: ", format(np.std(f1_list), '.4f'))