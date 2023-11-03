Instructions:

Processing data: 

After obtaining data from the Semeval Task organisers, store it in directory /data/

To process data for subtask 1, call the function produce_joined_data() from /st1/st1_utils_peft.py 
This will produce and save three tsv files:
- organizer train set (/st1/processed_data/train.tsv)
- organizer dev set (/st1/processed_data/dev.tsv)
- organizer train and dev (/st1/processed_data/joined.tsv)

(optional: obtaining translations)
joined.tsv can be uploaded to Google translate to obtain English translations. 
These translations can be added manually to the original joined.tsv under the new column 'text_en'.


Training: 

The three training scripts (run_trad_st1.py, run_lora_st1.py, run_pffeiffer_st1.py) correspond to the three settings explored in the paper: 
- full fine-tuning
- lora
- pfeiffer adapters. 

Each script will train 3 runs of the model, using the hyperparameters specified in the fixed_config dictionary. 