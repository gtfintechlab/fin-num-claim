import pandas as pd
import os

files = os.listdir('../data/grid_search_results')

files_xls = [f for f in files if '_bert-large.xlsx' in f]

for file in files_xls:
    df = pd.read_excel('../data/grid_search_results/' + file)
    df_temp = df.groupby(['Learning Rate', 'Batch Size'], as_index=False).agg(
    {
        "Val F1 Score": ["mean"],
        "ARPTS Test F1 Score": ["mean", "std"], 
        "EC Test F1 Score": ["mean", "std"],
        "EC Test Labeling Time(m)": ["mean"]
    }
    )
    df_temp.columns = ['Learning Rate', 'Batch Size', 'mean Val F1 Score', 'mean ARPTS Test F1 Score', 'std ARPTS Test F1 Score', 'mean EC Test F1 Score', 'std EC Test F1 Score', 'mean EC Test Labeling Time(m)']

    max_element = df_temp.iloc[df_temp['mean Val F1 Score'].idxmax()] 
    print(file)
    # print(max_element)
    # print(max_element['mean Test F1 Score'])
    print(format(max_element['mean ARPTS Test F1 Score'], '.4f'), "\n")
    print(format(max_element['std ARPTS Test F1 Score'], '.4f'), "\n")
    print(format(max_element['mean EC Test F1 Score'], '.4f'), "\n")
    print(format(max_element['std EC Test F1 Score'], '.4f'), "\n")
    print("Time in m", format(max_element['mean EC Test Labeling Time(m)'], '.4f'), "\n")