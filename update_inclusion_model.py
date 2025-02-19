import gspread
import json
import pandas as pd
import os

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, TrainingArguments, Trainer

# GLOBAL VARS
FILEPATH = '.'
GOOGLE_SPREADSHEET_ID = 'REAL' # 'REAL' OR 'TEST'
DATA_SHEET_NAME = 'data' # NAME OF DATA SHEET IN SPREADSHEET
LOCAL_DATA_RELPATH = '/data/second_gen/ratings.csv'
LOCAL_LOG_RELPATH = '/data/second_gen/extraction_log.csv'
INCLUSION_MODELS_TO_USE = {
    #
    # 'inclusion_biobert':{'relpath':'/models/production_models/inclusion_biobert','threshold_relpath':'/data/second_gen/thresholds_biobert.csv','threshold_sheet_name':'thresholds_biobert','model_source':'dmis-lab/biobert-base-cased-v1.2', 'lr':1e-5, 'epochs':4},
    'inclusion_biomedbert':{'relpath':'/models/production_models/inclusion_biomedbert','threshold_relpath':'/data/second_gen/thresholds_biomedbert.csv','threshold_sheet_name':'thresholds_biomedbert','model_source':'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract', 'lr':2e-5, 'epochs':3},
    } # DICT OF MODELS TO USE AND ASSOCIATED PARAMETERS

# FUNCTIONS

def force_update_google_credentials():
    credentials_filepath = FILEPATH + '/credentials/credentials.json'
    authorized_user_filepath = FILEPATH + '/credentials/authorized_user.json'
    if os.path.exists(authorized_user_filepath):
        os.remove(authorized_user_filepath)
    gc = gspread.oauth(
        credentials_filename = credentials_filepath,
        authorized_user_filename = authorized_user_filepath
    )
    return gc

def get_google_sheet(gc, google_spreadsheet_id, sheet_name):
    sht = gc.open_by_key(google_spreadsheet_id)
    sheet = sht.worksheet(sheet_name)
    return sheet

def get_end_date(local_log_relpath):
    extraction_log_df = pd.read_csv(FILEPATH + local_log_relpath, index_col=0).fillna('')
    end_date = extraction_log_df.iloc[-1]['date_end']
    return(end_date)

def get_inclusion_model_version(inclusion_model_relpath):
    version = max([int(x[1:]) for x in os.listdir(FILEPATH + inclusion_model_relpath)])
    return(version)

def update_threshold_local_data(local_threshold_relpath, end_date, new_threshold, thresholds, precisions, recalls, f1s, accuracies, excl_ratios, n_inc_excl):
    df_to_append = pd.DataFrame.from_dict([{'date_end':end_date, 'computed_threshold':new_threshold, 'thresholds':', '.join([str(t) for t in thresholds]), 'precisions': ', '.join([str(p) for p in precisions]), 'recalls':', '.join([str(r) for r in recalls]), 'f1s':', '.join([str(f) for f in f1s]), 'accuracies':', '.join([str(a) for a in accuracies]), 'exclusion_ratios':', '.join([str(r) for r in excl_ratios]), 'n_incorrectly_excluded':', '.join([str(i) for i in n_inc_excl])}])
    thresholds_df = pd.read_csv(FILEPATH + local_threshold_relpath, index_col=0).fillna('')
    updated_thresholds = pd.concat([thresholds_df, df_to_append], ignore_index=True)
    updated_thresholds.to_csv(FILEPATH + local_threshold_relpath)

def update_threshold_google_sheet(threshold_sheet, end_date, new_threshold, thresholds, precisions, recalls, f1s, accuracies, excl_ratios, n_inc_excl):
    threshold_sheet.append_row([end_date, new_threshold, ', '.join([str(t) for t in thresholds]), ', '.join([str(p) for p in precisions]), ', '.join([str(r) for r in recalls]), ', '.join([str(f) for f in f1s]), ', '.join([str(a) for a in accuracies]), ', '.join([str(r) for r in excl_ratios]), ', '.join([str(i) for i in n_inc_excl])])

def compute_new_threshold(local_data_relpath, inclusion_model_source, inclusion_model_lr, inclusion_model_epochs, tokenizer_function, tokenizer, tokenizer_kwargs):
    df = pd.read_csv(FILEPATH + local_data_relpath)[1:].fillna('')
    df['labels'] = df.apply(lambda x: x['rating_consensus'] if x['rating_consensus'] != '' else x['rating1'] if ((x['rating1'] == x['rating2']) and (x['consensus_reason'] == '') and (x['rating1'] != '')) else 'error', axis=1)
    df_columns_to_keep = ['text', 'labels']
    df = df[df_columns_to_keep]
    
    kf = KFold(n_splits=5, shuffle=True)
    
    thresholds = []
    precisions = []
    recalls = []
    f1s = []
    accuracies = []
    excl_ratios = []
    n_inc_excl = []

    n_fold=0
    for train_idx, val_idx in kf.split(df):
        n_fold+=1
        print('Strating model training for fold: {}'.format(n_fold))
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]
        ds = Dataset.from_pandas(train)
        ds.cleanup_cache_files()
        ds = ds.class_encode_column('labels')

        tokenized_ds = ds.map(tokenizer_function, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(inclusion_model_source, num_labels=2)
        training_args = TrainingArguments(FILEPATH + '/.temp', save_strategy='no', eval_strategy='no', logging_strategy='no', overwrite_output_dir=True, learning_rate=inclusion_model_lr, num_train_epochs=inclusion_model_epochs)
        trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_ds)
        trainer.train()

        print('Starting threshold calculation for fold: {}'.format(n_fold))
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None, device=0)
        scores = []
        for i in range(len(val)):
            scores.append(pipe(val.iloc[i]['text'], **tokenizer_kwargs)[0][1]['score'])

        pr,re,th = precision_recall_curve(val['labels'], scores)
        rth_df = pd.DataFrame.from_dict({'precision':pr[:-1],'recall':re[:-1],'threshold':th})

        ceiling_recall = 0.99
        threshold = rth_df[rth_df.recall >= ceiling_recall].iloc[-1]['threshold']
        thresholds.append(threshold)

        included = []
        preds = []
        incorrect = []
        for i, label, score in zip(range(len(val['labels'])), val['labels'], scores):
            if score >= threshold:
                included.append(i)
                preds.append(1)
            else:
                if label == 1:
                    incorrect.append(i)
                preds.append(0)
        precisions.append(precision_score(val['labels'], preds))
        recalls.append(recall_score(val['labels'], preds))
        f1s.append(f1_score(val['labels'], preds))
        accuracies.append(accuracy_score(val['labels'], preds))
        excl_ratios.append((len(val) - len(included)) / len(val))
        n_inc_excl.append(len(incorrect))

    thr_df = pd.DataFrame(thresholds)
    new_threshold = thr_df.mean()[0]
    print('Mean threshold: {:.20f}'.format(new_threshold))
    return(new_threshold, thresholds, precisions, recalls, f1s, accuracies, excl_ratios, n_inc_excl)

def update_inclusion_model(local_data_relpath, inclusion_model_source, inclusion_model_relpath, current_model_version, inclusion_model_lr, inclusion_model_epochs, tokenizer_function):
    new_model_version = int(current_model_version) + 1
    print('Training new version of inclusion model going from version {} to version {}'.format(current_model_version, new_model_version))

    df = pd.read_csv(FILEPATH + local_data_relpath)[1:].fillna('')
    df['labels'] = df.apply(lambda x: x['rating_consensus'] if x['rating_consensus'] != '' else x['rating1'] if ((x['rating1'] == x['rating2']) and (x['consensus_reason'] == '') and (x['rating1'] != '')) else 'error', axis=1)
    df_columns_to_keep = ['text', 'labels']
    df = df[df_columns_to_keep]

    ds = Dataset.from_pandas(df)
    ds.cleanup_cache_files()
    ds = ds.class_encode_column('labels')

    tokenized_ds = ds.map(tokenizer_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(inclusion_model_source, num_labels=2)
    for param in model.parameters(): param.data = param.data.contiguous() # LINE ADDED BECAUSE OF A WEIRD BUG THAT PREVENTS SAVING BIOMEDBERT MODELS BECAUSE OF NON CONTIGUOUS TENSORS
    training_args = TrainingArguments(FILEPATH + inclusion_model_relpath + '/v{}'.format(new_model_version), save_strategy='no', eval_strategy='no', logging_strategy='no', learning_rate=inclusion_model_lr, num_train_epochs=inclusion_model_epochs)
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_ds)
    trainer.train()
    trainer.save_model(FILEPATH + inclusion_model_relpath + '/v{}'.format(new_model_version))

# MAIN

if __name__ == '__main__':

    with open(FILEPATH + '/credentials/spreadsheet_ids.json', mode='r') as file:
        spreadsheet_ids = json.load(file)

    if GOOGLE_SPREADSHEET_ID == 'REAL':
        google_spreadsheet_id = spreadsheet_ids['real_google_spreadsheet_id']
    else:
        google_spreadsheet_id = spreadsheet_ids['test_google_spreadsheet_id']

    end_date = get_end_date(LOCAL_LOG_RELPATH)

    google_credentials = force_update_google_credentials()
    
    for inclusion_model_name, inclusion_model_parameters in INCLUSION_MODELS_TO_USE.items():
        print('\n\nPreparing inclusion model for update: {}'.format(inclusion_model_name))
        current_inclusion_model_version = get_inclusion_model_version(inclusion_model_parameters['relpath'])
        threshold_sheet = get_google_sheet(google_credentials, google_spreadsheet_id, inclusion_model_parameters['threshold_sheet_name'])
        tokenizer = AutoTokenizer.from_pretrained(inclusion_model_parameters['model_source'])
        def tokenizer_function(samples):
            return tokenizer(samples['text'], padding="max_length", truncation=True, max_length=512)
        tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
        #new_threshold, thresholds, precisions, recalls, f1s, accuracies, excl_ratios, n_inc_excl = compute_new_threshold(LOCAL_DATA_RELPATH, inclusion_model_parameters['model_source'], inclusion_model_parameters['lr'], inclusion_model_parameters['epochs'], tokenizer_function, tokenizer, tokenizer_kwargs)
        #update_threshold_local_data(inclusion_model_parameters['threshold_relpath'], end_date, new_threshold, thresholds, precisions, recalls, f1s, accuracies, excl_ratios, n_inc_excl)
        #update_threshold_google_sheet(threshold_sheet, end_date, new_threshold, thresholds, precisions, recalls, f1s, accuracies, excl_ratios, n_inc_excl)
        update_inclusion_model(LOCAL_DATA_RELPATH, inclusion_model_parameters['model_source'], inclusion_model_parameters['relpath'], current_inclusion_model_version, inclusion_model_parameters['lr'], inclusion_model_parameters['epochs'], tokenizer_function)
    print('Done !')