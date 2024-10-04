# Libraries
import pandas as pd
import numpy as np
import os
import re
import glob
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

gold_labelled_data_path = '../Dataset/GoldLabelledData/'

# WS model testing
def test_ws_model(test_file):
    y_true_all = []
    y_pred_all = []
    sentence = []
    y_true = []
    y_pred = []
    df_test = pd.read_csv(test_file)
    # Get the Manual annotated label and model generated label for complete GoldLabelled Data 
    for file in glob.glob(gold_labelled_data_path + "*.xlsx"):
        df = pd.read_excel(file)
        for i in range(len(df)):
            sentence.append(df['sentence'][i]) 
            y_true_all.append(df['Manual'][i])
            y_pred_all.append(df['Inclaim'][i])
    # Filter the required labels by matching text(sentence) 
    for i in range(len(df_test)):
        idx = sentence.index(df_test['text'][i])
        y_true.append(y_true_all[idx])
        y_pred.append(y_pred_all[idx])
    # Return accuracy 
    return accuracy_score(y_true, y_pred)


# Performance metrics for complete GoldLabelled Data evaluated after manual annotation
def test_overall_performance():
    manual = []
    model_label = []
    count=0
    for file_manual in glob.glob(gold_labelled_data_path + "*.xlsx"):
        df_train = pd.read_excel(file_manual)
        for i in range(len(df_train)):
            count+=1
            manual.append(df_train['Manual'][i])
            model_label.append(df_train['Inclaim'][i])
    print('\n Classification Report: \n', classification_report(manual, model_label))
    print('\n Confusion matrix: \n',confusion_matrix(manual, model_label))

def main():
    # Path of test file
    filename = '../Dataset/WS/testfile.csv'
    accuracy_value = test_ws_model(filename)
    print('\n Accuracy: ', accuracy_value)

    # Uncomment next line for overall perfomance metric of GoldLabelled data
    # test_overall_performance()

if __name__ == '__main__':
    main()
