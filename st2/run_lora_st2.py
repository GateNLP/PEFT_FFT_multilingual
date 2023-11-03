
import os
import st2_utils
import pandas as pd
from datasets import Dataset 
from st2_utils import load_df_from_tsv
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
from transformers import set_seed 



def objective(config):
    #load data from tsv
    train_df = load_df_from_tsv(config['path_to_train'], use_translations=config['training_lang'], convert_labels=True)
    train_df = train_df[['id', 'text', 'label', 'lang']]

    print(len(train_df))
    if config['en_only']: 
        train_df = st2_utils.holdout_lang(train_df, 'en')['held_out']

        print('After filtering',len(train_df))
    train_ds = Dataset.from_pandas(train_df)

    print(train_ds.select([0]))
    print('train:', len(train_ds))

    eval_df = load_df_from_tsv(config['path_to_eval'], use_translations='original', convert_labels=True)
    eval_df = eval_df[['id', 'text', 'label', 'lang']]
    eval_ds = Dataset.from_pandas(eval_df)
    print('org_test_set:', len(eval_ds))

    eval_indexes_lang_map = np.array(eval_ds['lang'])
    # print(type(eval_indexes_lang_map))
    # print(eval_indexes_lang_map)



    #tokenize 
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    def tokenization(example): 
        return tokenizer(example['text'], truncation=True, padding="max_length", max_length=config['max_length']) 
    train_ds = train_ds.map(tokenization, batched=True)
    eval_ds = eval_ds.map(tokenization, batched=True)

    def get_preds_from_logits(logits):
        ret = np.zeros(logits.shape)
        
        # we fill 1 to every class whose score is higher than some threshold
        # In this example, we choose that threshold = 0
        #with sigmoid function, threshold of 0 is equal to probability =0.5
        ret[:] = (logits[:] >= 0).astype(int)
        
        return ret

    def report_scores(labels, predictions, metric_group_prepend ='overall', dict_of_metrics = {}): 

        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro')
        # The global f1_metrics
        dict_of_metrics[metric_group_prepend+"_f1_micro"] = f1
        dict_of_metrics[metric_group_prepend+"_precision"] = precision 
        dict_of_metrics[metric_group_prepend+"_recall"] = recall 
        dict_of_metrics[metric_group_prepend+"_f1_macro"] = f1_score(labels, predictions, average="macro")
        if metric_group_prepend == 'overall': 
            f1_by_class = f1_score(labels, predictions, average=None).tolist()
            for i, f1 in enumerate(f1_by_class):
                curr_frame = str(i+1)
                dict_of_metrics[metric_group_prepend+'_f1_frame_'+curr_frame] = f1
        return dict_of_metrics


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        final_metrics = {}
        
        # Deduce predictions from logits
        predictions = get_preds_from_logits(logits)

        report_scores(labels,predictions,metric_group_prepend ='overall', dict_of_metrics =final_metrics)

        for language in ['en','fr','ge','it','po','ru']:
            mask = np.where(eval_indexes_lang_map == language, True,False)
            label_slice = labels[mask]
            prediction_slice = predictions[mask]
            report_scores(label_slice, prediction_slice, metric_group_prepend=language, dict_of_metrics = final_metrics)
            print(language, len(prediction_slice))

        return final_metrics


    model = AutoAdapterModel.from_pretrained(config['base_model'])
    model.add_classification_head('st2', num_labels=14)



    lora_layers_mlp = False 
    lora_layers_attn = False 

    if config['lora_layers'] == 'ffn' or config['lora_layers'] == 'all': 
        lora_layers_mlp = True 

    if config['lora_layers'] == 'attn' or config['lora_layers'] == 'all': 
        lora_layers_attn = True 

    my_adapter_config = LoRAConfig(
        r=config['lora_rank'],
        alpha=config['lora_rank'],
        selfattn_lora = lora_layers_attn, 
        intermediate_lora = lora_layers_mlp, 
        output_lora = lora_layers_mlp, 
        attn_matrices = config['lora_attn_matrices'],
        dropout = config['lora_dropout']
        )



    model.add_adapter("st2", config=my_adapter_config)

    model.train_adapter('st2')
    model.set_active_adapters('st2')

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", trainable_params)

    # Count non-trainable parameters
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Non-trainable parameters:", non_trainable_params)

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
        output_dir=config['output_dir'],
        learning_rate=config['learning_rate'],
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


        metric_for_best_model="overall_f1_micro",
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
    best_epoch = trainer.state.best_model_checkpoint
    return best, best_epoch 


def main():
    wandb.init(project='semeval-peft')
    score, epoch = objective(run_config)
    wandb.log({'best_f1_micro_score': score})
    wandb.log({'best_f1_micro_epoch': epoch})
    wandb.finish()


fixed_config = {
    'path_to_train': 'processed_data/clean_training_with_translations.tsv',
    'path_to_eval': 'processed_data/clean_dev_with_translations.tsv', 
    'training_lang': 'original', 

    'output_dir': 'models/lora',

    'en_only': False, 

    'base_model': 'xlm-roberta-large',
    'tokenizer_name': 'xlm-roberta-large', 
    'max_length': 512,
    'batch_size': 8, 
    'warmup_steps': 100, 
    'weight_decay': 0.01, 
    'num_train_epochs': 30, 


    #fixed params
    'lora_attn_matrices': ['q','k', 'v'], 
    'lora_dropout': 0.05,
    'lora_layers': 'all',
    'lora_rank': 8,
    'learning_rate': 3e-4
}


os.environ["WANDB_RUN_GROUP"] = fixed_config['output_dir']

#wandb sweep config  

for i in range(3): 
    run_config = fixed_config.copy()
    set_seed(i)
    run_config['output_dir'] = run_config['output_dir'] +'/'+'run_' + str(i)
    main()

