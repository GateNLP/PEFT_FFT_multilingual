from datasets import Dataset
import sys
import argparse
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoAdapterModel
from st3_utils import load_dataset, multibin, coarse_multibin
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
import requests
from bs4 import BeautifulSoup

# run = "base"
# run = "lora"
run = "adapter"

pretrained_name = "xlm-roberta-large"

with open(f'record_{run}.txt') as f:
    model_paths = f.read().splitlines()

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

    df.to_csv(folder_path + '/results.csv')

def tokenization(example): 
    return tokenizer(example['text'], truncation=True, padding="max_length", max_length=512, return_tensors="pt") 

# coarse = True
coarse = False
if __name__ == "__main__":
    now_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    langs = ["en", "fr", "ge", "it", "po", "ru", "gr", "ka", "es"]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for model_path in model_paths:
        if run == "base":
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=6 if coarse else 23)
        elif run == "lora":
            # checkpoint_reference = "/mnt/parscratch/users/acq22jal/tmp/semeval-ext/xlmr-lora/all_2_0.00015958290310372485/checkpoint-5688/st3/"
            model = AutoAdapterModel.from_pretrained(pretrained_name)
            adapter_name = model.load_adapter(model_path)
            model.set_active_adapters(adapter_name)
        elif run == "adapter":
            # checkpoint_reference = "/mnt/parscratch/users/acq22jal/tmp/semeval-ext/xlmr-pfeiffer/8_4.300697708306946e-05/checkpoint-6952/st3/"
            model = AutoAdapterModel.from_pretrained(pretrained_name)
            adapter_name = model.load_adapter(model_path)
            model.set_active_adapters(adapter_name)
        else:
            raise Exception("Wrong architecture type")


        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        print(f"Loaded checkpoint '{model_path}'")
        for lang in langs:
            print(lang)
            test_df = load_dataset([lang], split="test").set_index(["id", "line"])

            test_ds = Dataset.from_pandas(test_df).with_format("torch")
            test_ds = test_ds.map(tokenization, batched=True)
            # print(test_ds)
            dataloader = DataLoader(test_ds, batch_size=2, shuffle=False)

            full_pred = []
            for batch in dataloader:
                outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                logits = outputs[0]

                ret = np.zeros(logits.shape)
                ret[:] = (torch.sigmoid(logits)[:] >= 0.3).cpu().numpy().astype(int)

                full_pred.append(ret)

            preds = np.concatenate(full_pred)
            if coarse:
                out = coarse_multibin.inverse_transform(preds)
            else:
                out = multibin.inverse_transform(preds)
            out = list(map(lambda x: ",".join(x), out))
            out = pd.DataFrame(out, test_df.index)
            if os.path.exists(model_path + "/preds") == False:
                os.mkdir(model_path + "/preds")
            out.to_csv(
                f"{model_path}/preds/{lang}.txt",
                sep="\t",
                header=None,
            )

        save_point = model_path + '/preds'
        submit_from_folder(save_point)
    #averaging results

    model_paths_mod = [path +'/preds/results.csv' for path in model_paths]

    #change this range(4) value to however many sets of 3 you have. (e.g. I have 4 different model configurations, each one has been trained with 3 runs)
    for i in range(1):
        paths = [model_paths_mod[i*3], model_paths_mod[(i*3)+1], model_paths_mod[(i*3)+2]]
        dfs = [pd.read_csv(path) for path in paths]
        combined = pd.concat(dfs, axis = 1)


        combined['mean_f1_micro'] = combined.groupby(by=combined.columns, axis=1).apply(lambda g: g.mean(axis=1) if isinstance(g.iloc[0,0], float) else g.iloc[:,0])['f1-micro']
        combined['std_f1_micro_unbiased'] = combined.groupby(by=combined.columns, axis=1).apply(lambda g: g.std(axis=1, ddof = 1) if isinstance(g.iloc[0,0], float) else g.iloc[:,0])['f1-micro']
        combined['std_f1_micro_biased'] = combined.groupby(by=combined.columns, axis=1).apply(lambda g: g.std(axis=1, ddof=0) if isinstance(g.iloc[0,0], float) else g.iloc[:,0])['f1-micro']

        print(combined)
        common_prefix = os.path.commonpath(paths)
        combined.to_csv(common_prefix + '/avg_results.csv')
