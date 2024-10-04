import pandas as pd
import os

folder_out = "../data/SentiClaimLabelledData_no_filter/" 

filenames = [x for x in os.listdir(folder_out )]
url_df = pd.DataFrame({'filename':filenames})

res_list = []
column_list = ['file_name', 'positive', 'negative', 'neutral', 
               'numerical_positive', 'numerical_negative', 'numerical_neutral', 
               'numerical_financial_positive', 'numerical_financial_negative', 'numerical_financial_neutral', 
               'numerical_financial_inclaim_positive', 'numerical_financial_inclaim_negative', 'numerical_financial_inclaim_neutral']

for index, row in url_df.iterrows():
    if not index % 100:
        print(index)
    file_name = row['filename']
    full_filename = folder_out + file_name
    current_df = pd.read_excel(full_filename)

    if (len(current_df.index) > 0):

        temp_res_list = []

        # 'positive', 'negative', 'neutral',
        positive_sentences_df = current_df.loc[current_df['senti_label'] == "LABEL_2"]
        temp_res_list.append(positive_sentences_df.shape[0])

        negative_sentences_df = current_df.loc[current_df['senti_label'] == "LABEL_0"]
        temp_res_list.append(negative_sentences_df.shape[0])

        neutral_sentences_df = current_df.loc[current_df['senti_label'] == "LABEL_1"]
        temp_res_list.append(neutral_sentences_df.shape[0])
        

        # 'numerical_positive', 'numerical_negative', 'numerical_neutral',
        numerical_positive_sentences_df = positive_sentences_df.loc[positive_sentences_df['numerical'] == 1]
        temp_res_list.append(numerical_positive_sentences_df.shape[0])

        numerical_negative_sentences_df = negative_sentences_df.loc[negative_sentences_df['numerical'] == 1]
        temp_res_list.append(numerical_negative_sentences_df.shape[0])

        numerical_neutral_sentences_df = neutral_sentences_df.loc[neutral_sentences_df['numerical'] == 1]
        temp_res_list.append(numerical_neutral_sentences_df.shape[0])

        # 'numerical_financial_positive', 'numerical_financial_negative', 'numerical_financial_neutral', 
        numerical_financial_positive_sentences_df = numerical_positive_sentences_df.loc[numerical_positive_sentences_df['financial'] == 1]
        temp_res_list.append(numerical_financial_positive_sentences_df.shape[0])

        numerical_financial_negative_sentences_df = numerical_negative_sentences_df.loc[numerical_negative_sentences_df['financial'] == 1]
        temp_res_list.append(numerical_financial_negative_sentences_df.shape[0])

        numerical_financial_neutral_sentences_df = numerical_neutral_sentences_df.loc[numerical_neutral_sentences_df['financial'] == 1]
        temp_res_list.append(numerical_financial_neutral_sentences_df.shape[0])

        # 'numerical_financial_inclaim_positive', 'numerical_financial_inclaim_negative', 'numerical_financial_inclaim_neutral'
        numerical_financial_inclaim_positive_sentences_df = numerical_financial_positive_sentences_df.loc[numerical_financial_positive_sentences_df['Inclaim'] == 1]
        temp_res_list.append(numerical_financial_inclaim_positive_sentences_df.shape[0])

        numerical_financial_inclaim_negative_sentences_df = numerical_financial_negative_sentences_df.loc[numerical_financial_negative_sentences_df['Inclaim'] == 1]
        temp_res_list.append(numerical_financial_inclaim_negative_sentences_df.shape[0])

        numerical_financial_inclaim_neutral_sentences_df = numerical_financial_neutral_sentences_df.loc[numerical_financial_neutral_sentences_df['Inclaim'] == 1]
        temp_res_list.append(numerical_financial_inclaim_neutral_sentences_df.shape[0])

    else:
        temp_res_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    curr_row = list(row)
    curr_row = curr_row + temp_res_list
    res_list.append(curr_row)

result_df = pd.DataFrame(res_list, columns=column_list)
result_df.to_excel('../data/master_sentiment_AR_ablation_v2.xlsx', index=False)