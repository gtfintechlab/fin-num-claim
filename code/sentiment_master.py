import pandas as pd
import os

folder_out = "../data/SentiClaimLabelledData/" 

filenames = [x for x in os.listdir(folder_out )]
url_df = pd.DataFrame({'filename':filenames})

res_list = []
column_list = ['filename', 'inclaim_positive', 'inclaim_negative', 'inclaim_neutral']

for index, row in url_df.iterrows():
    if not index % 100:
        print(index)
    file_name = row['filename']
    full_filename = folder_out + file_name
    current_df = pd.read_excel(full_filename)

    if (len(current_df.index) > 0):

        temp_res_list = []

        positive_sentences_df = current_df.loc[current_df['senti_label'] == "LABEL_2"]
        temp_res_list.append(positive_sentences_df.shape[0])

        negative_sentences_df = current_df.loc[current_df['senti_label'] == "LABEL_0"]
        temp_res_list.append(negative_sentences_df.shape[0])

        neutral_sentences_df = current_df.loc[current_df['senti_label'] == "LABEL_1"]
        temp_res_list.append(neutral_sentences_df.shape[0])
    else:
        temp_res_list = [0, 0, 0]

    curr_row = list(row)
    curr_row = curr_row + temp_res_list
    res_list.append(curr_row)

result_df = pd.DataFrame(res_list, columns=column_list)
result_df.to_excel('../data/master_sentiment.xlsx', index=False)