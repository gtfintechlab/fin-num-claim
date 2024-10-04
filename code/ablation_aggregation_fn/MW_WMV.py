# Libraries
import pandas as pd
import numpy as np
import re
import spacy
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import MajorityLabelVoter,LabelModel
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
                return 1
    return -1

## High Confidence in-claim phrases
@labeling_function()
def phrases(x):
    list_of_phrases = ['aims to','to be ', 'likely to', 'intends to',' on track to ', ' pegged at ', ' earnings guidance to ']
    for word in list_of_phrases:
        if word in x.sentence:
            return 1
    return -1

## Low Confidence in-claim
@labeling_function()
def probable_phrases(x):
    list_of_phrases = ['to incur','touted to', 'entitled to']
    for word in list_of_phrases:
        if word in x.sentence:
            return 1
    return -1

## High Confidence out-of-claim(assertions and past-tense)
@labeling_function()
def past_tense_assertions(x):
    pattern = ["reasons to buy: ","reasons to sell: ","was ","were ","declares quarterly dividend ","last earnings report ","recorded "]
    for i in pattern:
        if re.search(i,x.sentence):
            return 0
    return -1

# Types of labelling functions included
lfs = [labels, phrases, probable_phrases,past_tense_assertions]
applier = PandasLFApplier(lfs=lfs)

for data_category in ["gold_arpts_numclaim-", "gold_ec_numclaim-"]:
    f1_list = []
    for seed in [5768, 78516, 944601]:
        
        # load and apply lfs on training and testing data
        df_train = pd.read_excel("../../data/train/" + data_category + "train-" + str(seed) + ".xlsx", engine='openpyxl')
        df_test = pd.read_excel("../../data/test/" + data_category + "test-" + str(seed) + ".xlsx", engine='openpyxl')

        df_train.columns = ["true_label", "sentence"]
        df_test.columns = ["true_label", "sentence"]
        if data_category == "gold_ec_numclaim-":
            df_train.columns = ["sentence", "true_label"]
            df_test.columns = ["sentence", "true_label"] 

        # Pass data for getting the labels
        L_train = applier.apply(df=df_train[["sentence"]])
        L_test = applier.apply(df=df_test[["sentence"]])

        status = []

        majority_model = MajorityLabelVoter(cardinality=2)
        label_model = LabelModel(cardinality=2, verbose=True)

        # # MAJORITY MODEL "Comment it when want to use label model"
        # preds_train = majority_model.predict(L=L_test)
        # # preds_train = np.where(preds_train == -1, 0, preds_train)

        # LABEL MODEL "Comment it when want to use majority model"
        label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=seed)
        preds_train = label_model.predict(L=L_test)
        # preds_train = np.where(preds_train == -1, 0, preds_train)

        f1_list.append(f1_score(df_test["true_label"], preds_train, average='weighted'))
    print(data_category + " f1 score mean: ", format(np.mean(f1_list), '.4f'))
    print(data_category + "f1 score std: ", format(np.std(f1_list), '.4f'))
