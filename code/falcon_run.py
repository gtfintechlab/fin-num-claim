import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd
import numpy as np
from time import time
from datetime import date


today = date.today()
seeds = [5768, 78516, 944601]

# set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str("0")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device assigned: ", device)

model = "tiiuae/falcon-7b-instruct"

# get model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)



for seed in [5768, 78516, 944601]: 

    # assign seed to numpy and PyTorch
    torch.manual_seed(seed)
    np.random.seed(seed)  

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


                prompts_list = []
                for i in range(len(sentences)): 
                    sen = sentences[i]
                    message = start_prompt[1] + few_shots[1] + "Now, for the following sentence provide the label in the first line and provide a short explanation in the second line. The sentence: " + sen
                    prompts_list.append(message)
                
                # documentation: https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation
                res = pipeline(
                    prompts_list,
                    max_new_tokens=512,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    )
                output_list = []

                for i in range(len(res)):
                    # print(res[i][0]['generated_text'][len(prompts_list[i]):])
                    output_list.append([labels[i], sentences[i], res[i][0]['generated_text'][len(prompts_list[i]):]])


                results = pd.DataFrame(output_list, columns=["true_label", "original_sent", "text_output"])

                time_taken = int((time() - start_t)/60.0)
                results.to_csv(f'../data/llm_prompt_output_feb_2024/falcon7binstruct_{few_shots[0]}_{start_prompt[0]}_{data_category}_{seed}_{today.strftime("%d_%m_%Y")}_{time_taken}.csv', index=False)
