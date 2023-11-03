
import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="5"

from st3_utils import load_dataset
import pandas as pd
from datasets import Dataset 
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding
import torch
import numpy as np
import wandb
from transformers.adapters import PfeifferConfig

from transformers import AdapterTrainer, AutoAdapterModel

def objective(sweep_config, config):
    #load data from tsv
    # train_df = load_df_from_tsv(config['path_to_train'], use_translations='original', convert_labels=True)
    train_df = load_dataset(['en','fr','ge','it','po','ru'], split="train", with_translations=False)
    train_ds = Dataset.from_pandas(train_df)

    print(train_ds.select([0]))
    print('train:', len(train_ds))


    eval_df = load_dataset(['en','fr','ge','it','po','ru'], split="dev", with_translations=False)
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
        logits = torch.from_numpy(logits)
        ret[:] = (torch.sigmoid(logits)[:] >= 0.3).numpy().astype(int)
        
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
    model.add_classification_head('st3', num_labels=23)


    my_adapter_config = PfeifferConfig(
        reduction_factor=sweep_config.reduction_factor
        )



    model.add_adapter("st3", config=my_adapter_config)

    model.train_adapter('st3')
    model.set_active_adapters('st3')

    class MyAdapterTrainer(AdapterTrainer):
        # def __init__(self, group_weights=None, **kwargs):
        #     super().__init__(**kwargs)
        #     self.group_weights = group_weights #to experiment with - weighting imbalanced classes
            
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs[0]
            
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())

            return (loss, outputs) if return_outputs else loss


    class PrinterCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, logs=None, **kwargs):
            print(f"Epoch {state.epoch}: ")

    training_args = TrainingArguments(
        output_dir=config['output_dir']+ '/' + str(sweep_config["reduction_factor"])+'_'+str(sweep_config['learning_rate']),
        learning_rate=sweep_config.learning_rate,
        per_device_train_batch_size=sweep_config['batch_size'],
        per_device_eval_batch_size=sweep_config['batch_size'],
        num_train_epochs=config['num_train_epochs'],
        warmup_steps =sweep_config['warmup_steps'], 
        optim = 'adamw_torch',

        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps = 1,
        save_strategy = 'epoch',
        save_total_limit = 2,

        #adamw weight decay
        weight_decay=sweep_config['weight_decay'],
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
    
    return best


def main():
    wandb.init(project='pfeiffer-sweep')
    score = objective(wandb.config, fixed_config)
    wandb.log({'best_f1_micro_score': score})


fixed_config = {
    'output_dir': 'models/xlmr-pfeiffer',

    'base_model': 'xlm-roberta-large',
    'tokenizer_name': 'xlm-roberta-large',
    'max_length': 512,
    'num_train_epochs': 10, 
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
        'learning_rate': {'max': 1e-5, 'min': 1e-9},
        'warmup_steps': {"max": 5000, "min": 0}, 
        'weight_decay': {"max": 0.1, "min": 1e-5}, 
        'batch_size': {"values": [8, 16, 32, 64]},
        'reduction_factor': {"values": [2, 4, 8, 16, 32]} 

    }


}


#3: Start the sweep
#sweep_id = wandb.sweep(
#    sweep=sweep_config, 
#    project='pfeiffer-sweep'
#    )

sweep_id = "i8oap158"
wandb.agent(sweep_id, function=main, count=20, project = 'pfeiffer-sweep')

