# Libraries
import pandas as pd
import numpy as np
import logging
import os
import re
import glob
import spacy
import random
from snorkel.labeling import labeling_function, PandasLFApplier
nlp = spacy.load('en_core_web_sm')


# Basic paths and universal variables
dictionaryFile = pd.read_excel("../Dataset/financial_dictionary.xlsx")
dictionaryData = dictionaryFile.Word
financialWordsSet = set(dictionaryData.unique())
raw_data_path = '../Dataset/rawData/'
preprocessed_data_path = '../Dataset/PreprocessedData/'
labelled_data_path = '../Dataset/LabelledData/'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Preprocessing
## Basic preprocessing
def split_into_sentences(text):
    
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|me|edu)"
    digits = "([0-9])"
    
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

## Numeric Filter
def check_for_numeric_data(text):
    dollars = [x[0] for x in re.findall('(\$[0-9]+(\.[0-9]+)?)', text)]
    percentage = re.findall('\d*%', text)
    if(len(dollars)> 0 or len(percentage)>0):
        return True
    else:
        return False

## Financial Filtering
def financial_filter_data(text):
    global financialWordsSet 
    for word in text.split():
        if word in financialWordsSet:
            return True
    return False


## Driver function for preprocessing data
def preprocess_Data():
    for file in glob.glob(raw_data_path + "*.txt"):
        idx = 0
        df1 = pd.DataFrame(columns=['sentence'])
        f = open(file, "r")
        f = f.read()
        tempPath = os.path.splitext(file)[0] 
        
        # Basic preprocessing
        text = split_into_sentences(f)
        for sentence in text:
            sentence = sentence.lower()
            # Blacklisting non-numeric sentence
            if(check_for_numeric_data(sentence)):
                # Whitelisting financially relevant sentences
                if financial_filter_data(sentence):
                    df1.loc[idx] = [sentence]
                    idx += 1
        name = tempPath.split("\\")[-1]
        filename = preprocessed_data_path + str(name) + ".xlsx"
        # Write preprocessed file
        df1.to_excel(filename)
    logging.info('Preprocessing completed.')



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

## Driver function for labelling data
def label_Data():
    for file_manual in glob.glob(preprocessed_data_path + "*.xlsx"):
        df_train = pd.read_excel(file_manual)

        # Types of labelling functions included
        lfs = [labels, phrases, probable_phrases,past_tense_assertions]
        applier = PandasLFApplier(lfs=lfs)

        # Pass data for getting the labels
        L_train = applier.apply(df=df_train)
        status = []

        # Aggregating function to decide the final label
        for i in range(len(L_train)):
            if -1 in L_train[i]:
                status.append(0)
            else:
                if np.max(L_train[i])>0:
                    status.append(1)
                else:
                    status.append(0)
        df_train['Inclaim'] = status
        df_train.to_excel(labelled_data_path + file_manual.split('\\')[-1])
    
    logging.info('Labelling completed.')


def main():
    preprocess_Data()
    label_Data()
    
if __name__ == '__main__':
    main()