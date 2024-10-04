# Libraries
import pandas as pd
import numpy as np
import re
import spacy
from snorkel.labeling import labeling_function, PandasLFApplier
nlp = spacy.load('en_core_web_sm')
from sklearn.metrics import f1_score, accuracy_score

# Labelling Functions

## High Confidence in-claim 
## 1) lemmatized word matching
## 2) POS-tagging
@labeling_function()
def labels(x):
    list_of_labels = ['expect', 'project','predict','forecast','anticipate','contemplate','envision']
    words = nlp(x.sentence)
    arr_lemma = [(token,token.lemma_) for token in words]
   
    for word in arr_lemma:
        if word[1] in list_of_labels:
            if (word[1]=='project' and (word[0].tag_ in ['VBN' ,'VB','VBD','VBG','VBP','VBZ'])) or word[1]!='project': 
                return 2
    return 0

## High Confidence in-claim phrases
@labeling_function()
def phrases(x):
    list_of_phrases = ['aims to','to be ', 'likely to', 'intends to',' on track to ', ' pegged at ', ' earnings guidance to ']
    for word in list_of_phrases:
        if word in x.sentence:
            return 2
    return 0

## Low Confidence in-claim
@labeling_function()
def probable_phrases(x):
    list_of_phrases = ['to incur','touted to', 'entitled to']
    for word in list_of_phrases:
        if word in x.sentence:
            return 1
    return 0

## High Confidence out-of-claim(assertions and past-tense)
@labeling_function()
def past_tense_assertions(x):
    pattern = ["reasons to buy: ","reasons to sell: ","was ","were ","declares quarterly dividend ","last earnings report ","recorded "]
    for i in pattern:
        if re.search(i,x.sentence):
            return -1
    return 0

# Types of labelling functions included
lfs = [labels, phrases, probable_phrases,past_tense_assertions]
applier = PandasLFApplier(lfs=lfs)


        
# load and apply lfs on training and testing data
df_test = pd.read_excel("../data/test/gold_arpts_non_author_annotation.xlsx", engine='openpyxl')
# df_test = df_test.drop(df_test[df_test['label_non_author'] == "DBT"].index)
df_test = df_test.dropna()
print(df_test.shape)
df_test = df_test[df_test.label_non_author.apply(lambda x: str(x).isnumeric())]
print(df_test.shape)

df_test.columns = ["our_true_label", "label_non_author", "sentence"]

L_test = applier.apply(df=df_test[["sentence"]])

status = []

# Aggregating function to decide the final label
for i in range(len(L_test)):
    if -1 in L_test[i]:
        status.append(0)
    else:
        if np.max(L_test[i])>0:
            status.append(1)
        else:
            status.append(0)
df_test['predict'] = status

df_test["label_non_author"] = df_test["label_non_author"].astype('int')
print(f1_score(df_test["label_non_author"], df_test['predict'], average='weighted'))
print(f1_score(df_test["our_true_label"], df_test['predict'], average='weighted'))


