
import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="5"

import pandas as pd
from datasets import Dataset 
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding
import torch
import numpy as np
import wandb
from transformers.adapters import LoRAConfig

from transformers import AdapterTrainer, AutoAdapterModel
from transformers.adapters import CompacterConfig, AdapterConfig, PfeifferConfig 
from datasets import concatenate_datasets

import json 




def objective(sweep_config, config):
    #st2 specific code removed: 
    #loading datasets: train_ds and eval_ds
    #define compute_metrics 


    #loading model
    model = AutoAdapterModel.from_pretrained(config['base_model'])
    model.add_classification_head('st2', num_labels=14) #hard coded



    lora_layers_mlp = False 
    lora_layers_attn = False 

    if sweep_config.lora_layers == 'ffn' or sweep_config.lora_layers == 'all': 
        lora_layers_mlp = True 

    if sweep_config.lora_layers == 'attn' or sweep_config.lora_layers == 'all': 
        lora_layers_attn = True 

    my_adapter_config = LoRAConfig(
        r=sweep_config.lora_rank, 
        alpha=sweep_config.lora_rank, 
        selfattn_lora = lora_layers_attn, 
        intermediate_lora = lora_layers_mlp, 
        output_lora = lora_layers_mlp, 
        attn_matrices = config['lora_attn_matrices'],
        dropout = config['lora_dropout']
        )



    model.add_adapter("st2", config=my_adapter_config)

    model.train_adapter('st2')
    model.set_active_adapters('st2')

    class MyAdapterTrainer(AdapterTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs[0]
            
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

            return (loss, outputs) if return_outputs else loss


    class PrinterCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, logs=None, **kwargs):
            print(f"Epoch {state.epoch}: ")

    training_args = TrainingArguments(
        output_dir=config['output_dir']+ '/' + str(sweep_config['lora_layers'])+'_'+ str(sweep_config['lora_rank']) +'_'+str(sweep_config['learning_rate']),
        learning_rate=sweep_config.learning_rate,
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        num_train_epochs=config['num_train_epochs'],
        warmup_steps =config['warmup_steps'], 
        optim = 'adamw_torch',

        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps = 1,
        save_strategy = 'epoch',
        save_total_limit = 2,

        #adamw weight decay
        weight_decay=config['weight_decay'],
        #default pytorch values for adamw betas: 0.9, 0.999


        metric_for_best_model="overall_f1_micro", #hard coded 
        load_best_model_at_end=True,
        report_to = ["wandb"],
        # run_name = config['output_dir']
    )

    trainer = MyAdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[PrinterCallback],
    )

    trainer.train()
    
    best = trainer.state.best_metric 
    
    return best


def main():
    wandb.init(project='lora-sweep')
    score = objective(wandb.config, fixed_config)
    wandb.log({'best_f1_micro_score': score})


fixed_config = {
    'path_to_train': 'processed_data/clean_training_with_translations.tsv',
    'path_to_eval': 'processed_data/clean_dev_with_translations.tsv', 

    'output_dir': 'models/xlmr-lora/',

    'base_model': 'xlm-roberta-large',
    'tokenizer_name': 'xlm-roberta-large', 
    'max_length': 512,
    'batch_size': 8, 
    'warmup_steps': 100, 
    'weight_decay': 0.01, 
    'num_train_epochs': 30, 

    #fixed params
    'lora_attn_matrices': ['q','k', 'v'], 
    'lora_dropout': 0.05
}


#wandb sweep config  

sweep_config  = {
    'method': 'random', 
    'name': 'sweep', 
    'metric':
    {
        'goal': 'maximize', 
        'name': 'best_f1_micro_score', 
    },
    'parameters':{
        'learning_rate': {'max': 1e-3, 'min': 1e-5},
        'lora_rank': {'values': [2,8,32,128]},
        'lora_layers': {'values': ['all']} #apply lora to all layers (alternatives would be 'ffn' and 'attn')

    }


}

#3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_config, 
    project='lora-sweep'
    )


wandb.agent(sweep_id, function=main, count=20, project = 'lora-sweep')

