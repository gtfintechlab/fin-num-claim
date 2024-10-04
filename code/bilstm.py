import os
import sys
from time import time
import pandas as pd
import torch
from torch import nn
import random 
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm 
from torchtext.legacy  import data
from torchtext import vocab
import re
import spacy
from spacy.tokenizer import Tokenizer
from torch.utils.data import DataLoader

nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)
# sys.path.append('..')


class GloveLSTM(torch.nn.Module):
    def __init__(self, vectors, num_labels, embedding_dim=300, hidden_dim=128, dropout=0.5):
        super(GloveLSTM, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(vectors, freeze=False)
        self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input, lengths=None):
        emb = self.embedding(input)
        assert(lengths is not None)
        output = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths.to('cpu'))
        lstm_out, (h_n, c_n) = self.lstm(output)
        hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim = 1)
        return self.fc(hidden)
    
def fine_tune_plm(gpu_numbers: str, train_data_path: str, seed: int, batch_size: int, learning_rate: float, save_model_path: str, debug: bool = False):
    """
    Description: Run experiment over particular batch size, learning rate and seed
    """
    #Stanza 
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_numbers)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if debug:
        print("Device assigned: ", device)

    # load training data
    data_df = pd.read_excel(train_data_path, engine='openpyxl')
    num_labels = 2
    sentences = data_df['text'].to_list()
    labels = data_df['label'].tolist()

    TEXT = data.Field(include_lengths=True, fix_length=150, tokenize=lambda x: [tok.text for tok in tokenizer(x)])
    LABEL = data.Field(sequential=False, use_vocab=False)
    fields=[('tweet', TEXT), ('label', LABEL)]
    dataset = data.Dataset([data.Example.fromlist([sentences[i], labels[i]], fields) for i in range(len(sentences))], fields)
    TEXT.build_vocab(dataset, vectors=vocab.GloVe(), max_size=10000)

    train, validation = dataset.split(split_ratio=[0.8, 0.2], random_state=random.seed(seed))
    train_iter, val_iter = data.BucketIterator.splits((train,validation), batch_sizes=(batch_size, batch_size),
                                              sort_key=lambda x: len(x.tweet),
                                              sort_within_batch=True,
                                              shuffle=True,
                                              device=device) 
    valLength = len(validation)
    dataloaders_dict = {'train': train_iter, 'val': val_iter}
    experiment_results = []

    # assign seed to numpy and PyTorch
    torch.manual_seed(seed)
    np.random.seed(seed) 
    # select optimizer
    model = GloveLSTM(TEXT.vocab.vectors, num_labels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    max_num_epochs = 100
    max_early_stopping = 7
    early_stopping_count = 0
    best_ce = float('inf')
    best_accuracy = float('-inf')
    best_f1 = float('-inf')
    eps = 1e-2
    criterion = torch.nn.CrossEntropyLoss().to(device)

    start_fine_tuning = time()

    for epoch in range(max_num_epochs):
        if (early_stopping_count >= max_early_stopping):
            break
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                early_stopping_count += 1
            else:
                model.eval()
            
            curr_ce = 0
            curr_accuracy = 0
            actual = torch.tensor([]).long().to(device)
            pred = torch.tensor([]).long().to(device)

            for batch in dataloaders_dict[phase]:
                optimizer.zero_grad()
                text, text_lengths = batch.tweet
                labels = batch.label 
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(text, text_lengths)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        curr_ce += loss.item() * text.size(1)
                        curr_accuracy += torch.sum(torch.max(outputs, 1)[1] == labels).item()
                        actual = torch.cat([actual, labels], dim=0)
                        pred= torch.cat([pred, torch.max(outputs, 1)[1]], dim=0)
            if phase == 'val':
                curr_ce = curr_ce / valLength
                curr_accuracy = curr_accuracy / valLength
                currF1 = f1_score(actual.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted')
                if currF1 >= best_f1 + eps:
                    best_f1 = currF1
                    best_accuracy = curr_accuracy
                    best_ce = curr_ce
                    early_stopping_count = 0
                    torch.save({
                                'model_state_dict': model.state_dict(),
                                }, 'best_model.pt')
                print("Val CE: ", curr_ce)
                print("Val Accuracy: ", curr_accuracy)
                print("Val F1: ", currF1)
                print("Early Stopping Count: ", early_stopping_count)
    training_time_taken = (time() - start_fine_tuning)/60.0

    ## ------------------testing---------------------
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    # test 1
    # load test data 1
    data_df_test = pd.read_excel("../data/test/gold_arpts_numclaim-test-" + str(seed) + ".xlsx", engine='openpyxl')
    sentences_test = data_df_test['text'].to_list()
    labels_test = data_df_test['label'].to_list()

    start_test_labeling = time()

    dataset_test = data.Dataset([data.Example.fromlist([sentences_test[i], labels_test[i]], fields) for i in range(len(sentences_test))], fields)

    test_iter = data.BucketIterator(dataset_test, batch_size=batch_size,
                                              sort_key=lambda x: len(x.tweet),
                                              sort_within_batch=True,
                                              shuffle=True,
                                              device=device) 
    dataloaders_dict_test = {'test': test_iter}
    experiment_results = []

    test_ce = 0
    test_accuracy = 0
    actual = torch.tensor([]).long().to(device)
    pred = torch.tensor([]).long().to(device)
    for batch in dataloaders_dict_test['test']:  
        optimizer.zero_grad()   
        text, text_lengths = batch.tweet
        labels = batch.label 
        with torch.no_grad():
            outputs = model(text, text_lengths)
            loss = criterion(outputs, labels)
            test_ce += loss.item() * text.size(1)
            test_accuracy += torch.sum(torch.max(outputs, 1)[1] == labels).item()
            actual = torch.cat([actual, labels], dim=0)
            pred = torch.cat([pred, torch.max(outputs, 1)[1]], dim=0)
    
    test_time_taken = (time() - start_test_labeling)/60.0
    test_ce = test_ce / len(dataset_test)
    test_accuracy = test_accuracy/ len(dataset_test)
    test_f1 = f1_score(actual.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted')
    print("Test F1: ", test_f1)
    experiment_results = [seed, learning_rate, batch_size, best_ce, best_accuracy, best_f1, training_time_taken, test_ce, test_accuracy, test_f1, test_time_taken]

    # load test data 2
    data_df_test = pd.read_excel("../data/test/gold_ec_numclaim-test-" + str(seed) + ".xlsx", engine='openpyxl')
    sentences_test = data_df_test['text'].to_list()
    labels_test = data_df_test['label'].to_list()

    start_test_labeling = time()

    dataset_test = data.Dataset([data.Example.fromlist([sentences_test[i], labels_test[i]], fields) for i in range(len(sentences_test))], fields)

    test_iter = data.BucketIterator(dataset_test, batch_size=batch_size,
                                              sort_key=lambda x: len(x.tweet),
                                              sort_within_batch=True,
                                              shuffle=True,
                                              device=device) 
    dataloaders_dict_test = {'test': test_iter}
    test_ce = 0
    test_accuracy = 0
    actual = torch.tensor([]).long().to(device)
    pred = torch.tensor([]).long().to(device)
    for batch in dataloaders_dict_test['test']:  
        optimizer.zero_grad()   
        text, text_lengths = batch.tweet
        labels = batch.label 
        with torch.no_grad():
            outputs = model(text, text_lengths)
            loss = criterion(outputs, labels)
            test_ce += loss.item() * text.size(1)
            test_accuracy += torch.sum(torch.max(outputs, 1)[1] == labels).item()
            actual = torch.cat([actual, labels], dim=0)
            pred = torch.cat([pred, torch.max(outputs, 1)[1]], dim=0)
    
    test_time_taken = (time() - start_test_labeling)/60.0
    test_ce = test_ce / len(dataset_test)
    test_accuracy = test_accuracy/ len(dataset_test)
    test_f1 = f1_score(actual.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted')
    experiment_results = experiment_results + [test_ce, test_accuracy, test_f1, test_time_taken]

    print(experiment_results)

    return experiment_results

 
def train_lm_experiments(gpu_numbers: str, train_data_path_prefix: str, language_model_to_use: str, data_category: str):
    """
    Description: Run experiments over different batch sizes, learning rates and seeds to find best hyperparameters
    """
    results = []
    seeds = [5768, 78516, 944601]
    batch_sizes = [32, 16, 8, 4]
    learning_rates = [1e-4, 1e-5, 1e-6, 1e-7]
    count = 0
    for i, seed in enumerate(seeds):
        for k, batch_size in enumerate(batch_sizes):
            for j, learning_rate in enumerate(learning_rates):

                count += 1
                print(f'Experiment {count} of {len(seeds) * len(batch_sizes) * len(learning_rates)}:')
                
                train_data_path = train_data_path_prefix + "-" + str(seed) + ".xlsx"

                results.append(fine_tune_plm(gpu_numbers, train_data_path, seed, batch_size, learning_rate, None, True))
                df = pd.DataFrame(results, columns=["Seed", "Learning Rate", "Batch Size", "Val Cross Entropy", "Val Accuracy", "Val F1 Score", "Fine Tuning Time(m)",
                                                    "ARPTS Test Cross Entropy", "ARPTS Test Accuracy", "ARPTS Test F1 Score", "ARPTS Test Labeling Time(m)",
                                                    "EC Test Cross Entropy", "EC Test Accuracy", "EC Test F1 Score", "EC Test Labeling Time(m)"])
                df.to_excel(f'../data/grid_search_results/train_{data_category}_{language_model_to_use}.xlsx', index=False)



if __name__=='__main__':
    start_t = time()

    # experiments
    for language_model_to_use in ["bi-lstm"]: # provide list of models
        data_category = "gold_arpts_numclaim"
        train_data_path_prefix = "../data/train/" + data_category + "-train"
        train_lm_experiments(gpu_numbers="0", train_data_path_prefix=train_data_path_prefix, language_model_to_use=language_model_to_use, data_category=data_category)
    
    
    print((time() - start_t)/60.0)