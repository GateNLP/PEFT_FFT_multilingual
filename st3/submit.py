import os
import numpy as np
import pandas as pd
import torch

import requests
from bs4 import BeautifulSoup

with open('record.txt') as f:
    model_paths = f.read().splitlines()

# #make predictions and save as .txt files
# PUT HERE - code that loads model and makes predictions on test set. Then saves predictions for each language in .txt file ('en.txt', 'ru.txt' etc.) 
# if you already have saved predictions, then this is not necessary

#submit predictions 
def submit_results(subtask: int, lang: str, results: str):
    if lang == "de":
        lang = "ge"
    
    assert lang in ["en", "it", "ru", "po", "fr", "ge", "es", "gr", "ka"]
    assert subtask in [1,2,3]

    files = {
        "sub": (f"{subtask}-{lang}.txt", results),
    }

    resp = requests.post(
        "https://propaganda.math.unipd.it/semeval2023task3/upload.php",
        data={
            "team": "vera",
            "passcode": "010266eee458bf67a1ebe380b083827e",
            "dataset": "dev",
            "task": f"{lang}{subtask}"
        },
        files=files,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
    }
    )

    bs = BeautifulSoup(resp.text, "html.parser")
    body_text = bs.body.get_text('\n')
    if 'Prediction file format is correct' not in body_text:
        print("Error from submission server:")
        print(body_text)
        print("=================")
        return

    _, f1_micro, f1_macro = tuple([td.text for td in bs.find('table').find_all('tr')[1].find_all('td')])
    f1_micro = float(f1_micro)
    f1_macro = float(f1_macro)

    return f1_micro, f1_macro

def submit_from_folder(folder_path):
    languages = ["en", "fr", "ge", "it", "po","ru", "es", "gr", "ka"]

    micros = []

    macros = []

    for lang in languages:

        path = folder_path + '/' + lang +'.txt'

        with open(path,"r") as f:

            read_results = f.read()

        mic, mac = submit_results(3, lang, read_results)

        micros.append(mic)

        macros.append(mac)

    my_dict = {

    'lang':languages,

    'f1-micro':micros,

    'f1-macro':macros

    }

    df = pd.DataFrame(my_dict)

    df.to_csv(folder_path + '/results_redo.csv')

for path_to_checkpoint in model_paths: 
    save_point = path_to_checkpoint + '/preds'
    submit_from_folder(save_point)


#averaging results

model_paths_mod = [path +'/preds/results_redo.csv' for path in model_paths]

#change this range(4) value to however many sets of 3 you have. (e.g. I have 4 different model configurations, each one has been trained with 3 runs)
for i in range(4):
    paths = [model_paths_mod[i*3], model_paths_mod[(i*3)+1], model_paths_mod[(i*3)+2]]
    dfs = [pd.read_csv(path) for path in paths]
    combined = pd.concat(dfs, axis = 1)


    combined['mean_f1_micro'] = combined.groupby(by=combined.columns, axis=1).apply(lambda g: g.mean(axis=1) if isinstance(g.iloc[0,0], float) else g.iloc[:,0])['f1-micro']
    combined['std_f1_micro_unbiased'] = combined.groupby(by=combined.columns, axis=1).apply(lambda g: g.std(axis=1, ddof = 1) if isinstance(g.iloc[0,0], float) else g.iloc[:,0])['f1-micro']
    combined['std_f1_micro_biased'] = combined.groupby(by=combined.columns, axis=1).apply(lambda g: g.std(axis=1, ddof=0) if isinstance(g.iloc[0,0], float) else g.iloc[:,0])['f1-micro']

    print(combined)
    common_prefix = os.path.commonpath(paths)
    combined.to_csv(common_prefix + '/avg_results.csv')