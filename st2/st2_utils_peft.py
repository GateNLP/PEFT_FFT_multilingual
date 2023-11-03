import pandas as pd
from tqdm import tqdm
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import classification_report as report
from sklearn.feature_extraction.text import CountVectorizer
import argparse
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from transformers import BertTokenizerFast, XLMRobertaTokenizerFast 
# from enc_t5 import EncMT5ForSequenceClassification, EncT5Tokenizer 

from ast import literal_eval 
import re 

import emoji 

from sklearn.model_selection import StratifiedKFold
import numpy as np

class LabelBinarizer(): 
    def __init__(self, *, pathtolabelmap = './processed_data/frame_map.tsv'):
        self.pathtolabelmap = pathtolabelmap
        #load frame map 
        self.df_framelabels = pd.read_csv(pathtolabelmap, sep = '\t', header = 0, names = ['frame_number', 'frame_label'])
        #note the first label 'Economic' has an index of 0 but a frame_number of 1 (frame_number is 1-indexed, so Economic is frame_1)  

    #string ('Economic, Health, Jurisprudence') to binary array (e.g. [1,0,0,1,0...,1,0,0]).
    def frame_to_label(self, frame_string: str): 
        labels = [0] * len(self.df_framelabels)

        for index, row in self.df_framelabels.iterrows(): 
            if row['frame_label'] in frame_string: 
                labels[index] = 1 
            else: 
                labels[index] = 0 
        
        return labels 
        


    #returns frames as array of strings
    def label_to_frame(self, label_arrray): 
        frames = []
        if type(label_arrray) == float:
            print("WARNING FLOAT")
            print(label_arrray)
        for idx, val in enumerate(label_arrray): 
            if val == 1: 
                frames.append(self.df_framelabels['frame_label'].iloc[idx])

        return frames 

#loads from tsv, adds a column 'label' (vector form of the string of frames)
def load_df_from_tsv(pathtodata, use_translations = 'original', convert_labels = True, merge_labels = False):     
    df = pd.read_csv(pathtodata, sep = '\t')

    if use_translations != 'original': 
        print('Replacing original langauge with translations in ', use_translations)
        print('Before: ', df.iloc[0])

        col_name = 'text_' + use_translations

        df['text'] = df[col_name]
        print('After: ', df.iloc[0])


    if convert_labels: 
        encoder = LabelBinarizer()

        if 'frames' in df.columns: 
            df['label']= df['frames'].apply(encoder.frame_to_label)
    return df 
    
def load_data(pathtodata, basemodel, maxlength, convert_labels = True, use_translations = 'original', truncation_side = 'right'): 
    
    #load data 
    df = load_df_from_tsv(pathtodata, use_translations, convert_labels)

    print('Loaded dataframe. Number of entries: ', df.shape[0])


    #load tokenizer - probably the better way to do this would be to pass the instantiated tokenizer... oh well
    if 'google/mt5' in basemodel: 
        tokenizer = EncT5Tokenizer.from_pretrained("google/mt5-base", truncation_side=truncation_side)
    elif 'xlm-roberta' in basemodel:
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(basemodel, truncation_side=truncation_side)
    else:  
        tokenizer = BertTokenizerFast.from_pretrained(basemodel, truncation_side=truncation_side)

    #construct preprocess function
    def tokenize_function(text_col):
        tokenized = tokenizer(text_col, truncation=True, padding="max_length", max_length=maxlength) 
        return tokenized 

    df['input_ids'] = df['text'].apply(lambda x: tokenize_function(x)['input_ids'])
    df['attention_mask'] = df['text'].apply(lambda x: tokenize_function(x)['attention_mask'])

    return df


#returns dataframe with two columns -> pred_labels (array of 1 or 0), pred_frames (array of frames)
def logits_to_preds(logits, df_inference): 
    predictions = (logits[:]>=0).astype(int)
    # print('Predictions from logits')
    # print(predictions)

    encoder = LabelBinarizer()

    df = pd.DataFrame()

    df["pred_labels"] = predictions.tolist()  
#     print('just labels')
#     print(df)
    
    df["pred_frames"] = df["pred_labels"].apply(encoder.label_to_frame)
#     print('with frames')
#     print(df)
    
    return df 


def make_dataframe(input_folder, labels_folder=None):
    #MAKE TXT DATAFRAME
    text = []
    
    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(input_folder))):
        print(fil)
        print(input_folder+fil)
        iD, txt = fil[7:].split('.')[0], open(input_folder +fil, 'r', encoding='utf-8').read()
        print(txt)
        text.append((iD, txt))

    df_text = pd.DataFrame(text, columns=['id','text']).set_index('id')
    df = df_text
    
    #MAKE LABEL DATAFRAME
    if labels_folder:
        labels = pd.read_csv(labels_folder, sep='\t', header=None)
        labels = labels.rename(columns={0:'id',1:'frames'})
        labels.id = labels.id.apply(str)
        labels = labels.set_index('id')

        #JOIN
        df = labels.join(df_text)[['text','frames']]
        
    
    return df


def produce_joined_data(path_to_data_folder = 'data/', save_location = 'st2/processed_data/'): 
    joined_df = pd.DataFrame(columns = ['id', 'lang', 'text', 'frames', 'dataset_origin'])
    for lang in ['en', 'fr', 'ge', 'it', 'po', 'ru']:
        for dataset_origin in ['train', 'dev']: 
            suffix = '-subtask-2'

            articles_path = path_to_data_folder + lang + '/' + dataset_origin + '-articles' + suffix + '/'
            labels_path = path_to_data_folder + lang + '/' + dataset_origin + '-labels' + suffix + '.txt'
            df = make_dataframe(articles_path, labels_path)
            df['dataset_origin'] = df['text'].apply(lambda x: dataset_origin)
            df['lang'] = df['text'].apply(lambda x: lang)

            joined_df = pd.concat([joined_df, df], ignore_index = True)
    
    save_path = save_location + 'joined.tsv'
    joined_df.to_csv(save_path, sep = '\t')


def holdout_lang(df, lang = 'it'): 
    should_holdout = df['lang'] == lang
    
    df_holdout = df[should_holdout]
    df_filtered = df[~should_holdout]

    

    return {'held_out': df_holdout, 'filtered': df_filtered}
