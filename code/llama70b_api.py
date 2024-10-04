import os,sys
import pandas as pd
from time import sleep, time
from datetime import date
today = date.today()
import numpy as np
from openai import OpenAI

TOGETHER_API_KEY = os.environ.get("")

# get date and api key
today = date.today()
# openai.api_key = APIKeyConstants.OPENAI_API_KEY

client = OpenAI(api_key="",
  base_url='https://api.together.xyz',
)

for seed in [5768, 78516, 944601]:  
    for data_category in ["gold_ec_numclaim-test-", "gold_arpts_numclaim-test-"]:
        start_t = time()
        # load training data
        test_data_path = "../data/test/" + data_category + str(seed) + ".xlsx"
        data_df = pd.read_excel(test_data_path)

        # data_df = data_df.head(1)

        sentences = data_df['text'].to_list()
        labels = data_df['label'].to_numpy()

        for start_prompt in [["simple", "Discard all the previous instructions. Behave like you are an expert sentence classifier. Classify the following sentence into 'INCLAIM', or 'OUTOFCLAIM' class. Label 'INCLAIM' if consist of a claim and not just factual past or present information, or 'OUTOFCLAIM' if it has just factual past or present information. "],
                             ["complex", "Discard all the previous instructions. Behave like you are an expert sentence classifier. Classify the following sentence into either 'INCLAIM' or 'OUTOFCLAIM'. 'INCLAIM' refers to predictions or expectations about financial outcomes. 'OUTOFCLAIM' refers to sentences that provide numerical information or established facts about past financial events. For each classification, 'INCLAIM' can be thought of as 'financial forecasts', and 'OUTOFCLAIM' as 'established financials'. "]]:
            for few_shots in [["zero", ""], # zero shot
                              ["few", "Here are a few examples: \
                                \nExample 1: free cash flow of $2.3 billion was up 10.5%, benefiting from the positive year-over-year change in net working capital due to covid at both nbcu and sky, half of which resulted from the timing of when sports rights payments were made versus when sports actually aired and half of which resulted from a slower ramp in content production. // The sentence is OUTOFCLAIM\
                                \nExample 2: we've also used our scale of more than 15,000 combined stores to drive merchandise cost savings exceeding $70 million. // The sentence is OUTOFCLAIM\
                                \nExample 3: consolidated total capital was $2.9 billion for the quarter. // The sentence is OUTOFCLAIM\
                                \nExample 4: third, as a result of the continued strength of the u.s. dollar, we are now factoring in an incremental fx headwind of $175 million across q3 and q4 revenue. // The sentence is INCLAIM\
                                \nExample 5: though early, we are planning our business based on the expectation of cy '23 wfe declining approximately 20% based on increasing global macroeconomic concerns and recent public statements from several customers, particularly in memory, and the impact of the new u.s. government regulations on native china investment.  // The sentence is INCLAIM\
                                \nExample 6: we expect revenue growth to be in the range of 5.5% to 6.5% year on year. // The sentence is INCLAIM "]]: # 6-shot


                output_list = []
                for i in range(len(sentences)): 
                    sen = sentences[i]
                    message = start_prompt[1] + few_shots[1] + "Now, for the following sentence provide the label in the first line and provide a short explanation in the second line. The sentence: " + sen

                    prompt_json = [
                            {"role": "user", "content": message},
                    ]
                    try:
                        chat_completion = client.chat.completions.create(
                        model="togethercomputer/llama-2-70b-chat",
                        messages=prompt_json,
                        temperature=0.0,
                        max_tokens=200
                        )
                    except Exception as e:
                        print(e)
                        i = i - 1
                        sleep(10.0)

                    answer = chat_completion.choices[0].message.content
                    
                    output_list.append([labels[i], sen, answer])
                    sleep(0.01) 

                results = pd.DataFrame(output_list, columns=["true_label", "original_sent", "text_output"])

                time_taken = int((time() - start_t)/60.0)
                results.to_csv(f'../data/llm_prompt_output_feb_2024/llama70b_{few_shots[0]}_{start_prompt[0]}_{data_category}_{seed}_{today.strftime("%d_%m_%Y")}_{time_taken}.csv', index=False)
