import gspread
import json
import math
import os
import pandas as pd
import requests
import time

from bs4 import BeautifulSoup 
from collections import defaultdict
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# GLOBAL VARS
FILEPATH = '.'
GOOGLE_SPREADSHEET_ID = 'REAL' # 'REAL' OR 'TEST'
DATA_SHEET_NAME = 'data' # NAME OF DATA SHEET IN SPREADSHEET
LOG_SHEET_NAME = 'extraction_log' # NAME OF LOG SHEET IN SPREADSHEET
LOCAL_LOG_RELPATH = '/data/second_gen/extraction_log.csv' # RELATIVE PATH TO LOCAL EXTRACTION LOG
LOCAL_THRESHOLD_RELPATH = '/data/second_gen/thresholds.csv' # RELATIVE PATH TO LOCAL THRESHOLD LOG
INCLUSION_MODEL_RELPATH = '/models/production_models/inclusion_biobert' # RELATIVE PATH TO PRODUCTION INCLUSION MODELS (NOT A SPECIFIC VERSION)
ORIGINAL_START_DATE = '2021/11/07' # FORMAT 'YYYY/MM/DD'
START_DATE = '2024/06/30' # FORMAT 'YYYY/MM/DD'
END_DATE = '2024/07/13' # FORMAT 'YYYY/MM/DD'
SEARCH_QUERY = 'pharmacists[All Fields] OR pharmacist[All Fields] OR pharmacy[title]' # PUBMED QUERY STRING
ABSTRACT_SECTIONS_TO_EXCLUDE = ['DISCLAIMER'] # List of abstract labels that will be excluded from data 

# FUNCTIONS

def get_google_sheet(google_spreadsheet_id, data_sheet_name):
    credentials_filepath = FILEPATH + '/credentials/credentials.json'
    authorized_user_filepath = FILEPATH + '/credentials/authorized_user.json'
    try:
        gc = gspread.oauth(
            credentials_filename = credentials_filepath,
            authorized_user_filename = authorized_user_filepath
        )
        sht = gc.open_by_key(google_spreadsheet_id)
        data_sheet = sht.worksheet(data_sheet_name)
    except:
        if os.path.exists(authorized_user_filepath):
            os.remove(authorized_user_filepath)
        gc = gspread.oauth(
            credentials_filename = credentials_filepath,
            authorized_user_filename = authorized_user_filepath
        )
        sht = gc.open_by_key(google_spreadsheet_id)
        data_sheet = sht.worksheet(data_sheet_name)
    return sht

def get_model_version(inclusion_model_relpath):
    version = max([int(x[1:]) for x in os.listdir(FILEPATH + inclusion_model_relpath)])
    print('Will use inclusion model version: {}'.format(version))
    return(version)

def get_exclusion_threshold(local_threshold_relpath):
    threshold_df = pd.read_csv(FILEPATH + local_threshold_relpath).fillna('')
    threshold = threshold_df.iloc[-1]['computed_threshold'].astype(float)
    print('Will use an exclusion threshold of: {:.5f}'.format(threshold))
    return(threshold)

def get_n_results(pubmed_credentials, search_query, start_date, end_date):
    params = {'db':'pubmed', 'term':search_query, 'datetype':'edat', 'mindate':start_date, 'maxdate':end_date, 'tool':pubmed_credentials['pubmed_tool_name'], 'email':pubmed_credentials['pubmed_tool_email']}
    r = requests.get(url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi', params=params)
    soup = BeautifulSoup(r.content, 'xml')
    n_results = soup.find('Count').text
    print('Results for current extraction: {}'.format(n_results))
    return(n_results)

def get_previous_pmids(local_log_relpath):
    extraction_log_df = pd.read_csv(FILEPATH + local_log_relpath, index_col=0).fillna('')
    extraction_log_df['pmids'] = extraction_log_df['pmids'].map(lambda x:x.split(', '))
    previous_pmids = []
    for l in extraction_log_df['pmids']:
        previous_pmids.extend(l)
    return(previous_pmids)

def build_pmid_list(n_results, previous_pmids, pubmed_credentials, search_query, original_start_date, start_date, end_date):
    current_pmids = []
    n_calls = math.ceil(int(n_results) / 100000)
    for i in tqdm(range(n_calls)):
        params = {'db':'pubmed', 'term':search_query, 'datetype':'edat', 'mindate':start_date, 'maxdate':end_date, 'retstart':i*100000, 'retmax':100000, 'tool':pubmed_credentials['pubmed_tool_name'], 'email':pubmed_credentials['pubmed_tool_email']}
        r = requests.get(url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi', params=params)
        results = BeautifulSoup(r.content, 'xml')
        pmids_call = results.find_all('Id')
    for id in pmids_call:
        current_pmids.append(id.text)
    time.sleep(0.35)

    potentially_missed_pmids = []
    n_calls = math.ceil(int(len(previous_pmids)) / 100000)
    for i in tqdm(range(n_calls)):
        params = {'db':'pubmed', 'term':search_query, 'datetype':'edat', 'mindate':original_start_date, 'maxdate':end_date, 'retstart':i*100000, 'retmax':100000, 'tool':pubmed_credentials['pubmed_tool_name'], 'email':pubmed_credentials['pubmed_tool_email']}
        r = requests.get(url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi', params=params)
        results = BeautifulSoup(r.content, 'xml')
        pmids_call = results.find_all('Id')
    for id in pmids_call:
        potentially_missed_pmids.append(id.text)
    time.sleep(0.35)

    missed_pmids = []
    for pmid in potentially_missed_pmids:
        if (pmid in previous_pmids) or (pmid in current_pmids):
            pass
        else:
            missed_pmids.append(pmid)

    print('Number of previously missed PMIDs: {}'.format(len(missed_pmids)))

    pmids = []
    for pmid in current_pmids:
        if (pmid in previous_pmids) or (pmid in missed_pmids):
            pass
        else:
            pmids.append(pmid)  
    print('Number of PMIDs retrieved for current extraction: {}'.format(len(pmids)))

    pmids.extend(missed_pmids)
    print('Number of PMIDs that will be retrieved: {}'.format(len(pmids)))
    return(pmids)

def retrieve_pubmed_data(pmids, pubmed_credentials):
    dataset = defaultdict()
    for pmid in tqdm(pmids):
        dataset[pmid] = defaultdict()
        params = {'db':'pubmed', 'id':pmid, 'retmode':'xml', 'tool':pubmed_credentials['pubmed_tool_name'], 'email':pubmed_credentials['pubmed_tool_email']}
        r = requests.get(url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', params=params)
        try:
            dataset[pmid]['pmdata'] = str(BeautifulSoup(r.content, 'xml'))
        except:
            dataset[pmid]['pmdata'] = None
            print('ERROR on call for pmid {}'.format(pmid))
        time.sleep(0.4)
    return dataset

def verify_pubmed_retrieval(ds):
    print('Length of dataset: {}'.format(len(ds)))
    n_pubmed = 0
    n_none = 0
    error_pmids = []
    for pmid, data in ds.items():
        if data != None:
            n_pubmed += 1
        else:
            n_none += 1
            error_pmids.append(pmid)
    print('Number of successfully retrieved PubMed sets: {}'.format(n_pubmed))
    print('Number of errors: {}'.format(n_none))
    if n_none > 0:
        print('ERROR in PubMed retrieval for following PMIDs, verify data: {}'.format(error_pmids))
        quit()

def build_text_and_filter_dataset(ds, abstract_sections_to_exclude):
    for pmid, data in ds.items():
        title = ''
        text = ''
        titleandtext = ''
        if data['pmdata'] == None:
            ds[pmid]['text'] = ''
        else:
            element_pmdata = BeautifulSoup(data['pmdata'], 'xml')
            pubyear = []
            try:
                for pubdate in element_pmdata.find_all('PubDate'):
                    pubyear.append(pubdate.find('Year').get_text())
            except:
                pass
            if len(pubyear) == 0:
                ds[pmid]['pubyear'] = None
            else:
                ds[pmid]['pubyear'] = pubyear[0]
            try:
                t = element_pmdata.find_all('ArticleTitle')
                title = [e.get_text() for e in t][0]
                if title == None:
                    title = ''
            except:
                title = ''
            if element_pmdata.find('Abstract') == None:
                labelsandtext = ''
            else:
                labels = [e['Label'] if 'Label' in e.attrs.keys() else '' for e in element_pmdata.find('Abstract').find_all('AbstractText')]
                text = [e.get_text() for e in element_pmdata.find('Abstract').find_all('AbstractText')]
                labelsandtext = ' '.join([' '.join([l,t]) for l,t in zip(labels, text) if l not in abstract_sections_to_exclude])
            titleandtext = title + ' ' + labelsandtext  
            ds[pmid]['text'] = titleandtext

    print('Number of elements in dataset: {}'.format(len(ds)))
    text_data_available = 0
    for _,data in ds.items():
        if data['text'] == '' or data['text'] == ' ':
            continue
        else:
            text_data_available += 1
    print('Number of elements with text data: {}'.format(text_data_available))

    filtered_dataset = defaultdict()
    for key,data in ds.items():
        if data['text'] == '' or data['text'] == ' ':
            continue
        else:
            filtered_dataset[key] = data
    print('Number of elements in filtered dataset: {}'.format(len(filtered_dataset)))
    
    return filtered_dataset

def make_inclusion_predictions(data, exclusion_threshold, model_version, inclusion_model_relpath):
    df = pd.DataFrame.from_dict(data, orient='index', columns=['text'])
    ds = Dataset.from_pandas(df)
    ds.cleanup_cache_files()
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
    model = AutoModelForSequenceClassification.from_pretrained(FILEPATH + inclusion_model_relpath + '/v{}'.format(model_version), local_files_only=True)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=0)
    scores = []
    for i in range(len(ds)):
        scores.append(pipe(ds[i]['text'], **tokenizer_kwargs)[0][1]['score'])
    df['inclusion_score'] = scores
    df['inclusion_suggestion'] = df['inclusion_score'].apply(lambda x: 'Exclude' if x < exclusion_threshold else 'Review')
    df['inclusion_model_version'] = model_version
    df = df.drop('inclusion_score', axis=1)
    df['rating1'] = ''
    df['rating2'] = ''
    df['rating_consensus'] = df.apply(lambda x: 0 if x['inclusion_suggestion'] == 'Exclude' else '', axis=1)
    df['consensus_reason'] = df.apply(lambda x: 'Excluded by ML model' if x['inclusion_suggestion'] == 'Exclude' else '', axis=1)
    n_filtered = len(df[df['inclusion_suggestion']=='Exclude'])
    n_included = len(df[df['inclusion_suggestion']=='Review'])
    print('Number of elements excluded by inclusion model: {}'.format(n_filtered))
    print('Number of elements to review: {}'.format(n_included))
    return(df)

def convert_df_to_rows (df):
    rows_to_append = df.reset_index().rename({'index':'PMID'},axis='columns').values.tolist()
    return rows_to_append

def update_local_data(pmids, start_date, end_date, exclusion_threshold, local_log_relpath):
    df_to_append = pd.DataFrame.from_dict([{'date_begin':start_date,'date_end':end_date,'n_results':len(pmids),'pmids':', '.join(pmids), 'exclusion_threshold':exclusion_threshold}])
    extraction_log_df = pd.read_csv(FILEPATH + local_log_relpath, index_col=0).fillna('')
    updated_extraction_log = pd.concat([extraction_log_df, df_to_append], ignore_index=True)
    updated_extraction_log.to_csv(FILEPATH + local_log_relpath)

def update_google_sheet(sht, data_sheet_name, log_sheet_name, rows_to_append, start_date, end_date, pmids, exclusion_threshold):
    data_sheet = sht.worksheet(data_sheet_name)
    data_sheet.batch_clear(['data_contents'])
    data_sheet.append_rows(rows_to_append)
    data_sheet.hide_columns(9,11)
    data_sheet.hide_columns(12,15)
    data_sheet.hide_columns(16,17)
    log_sheet = sht.worksheet(log_sheet_name)
    log_sheet.append_row([start_date, end_date, len(pmids), ', '.join(pmids), exclusion_threshold])

# MAIN

if __name__ == '__main__':

    with open(FILEPATH + '/credentials/spreadsheet_ids.json', mode='r') as file:
        spreadsheet_ids = json.load(file)

    if GOOGLE_SPREADSHEET_ID == 'REAL':
        google_spreadsheet_id = spreadsheet_ids['real_google_spreadsheet_id']
    else:
        google_spreadsheet_id = spreadsheet_ids['test_google_spreadsheet_id']
    
    with open(FILEPATH + '/credentials/pubmed_credentials.json', mode='r') as file:
        pubmed_credentials = json.load(file)

    google_sheet = get_google_sheet(google_spreadsheet_id, DATA_SHEET_NAME)
    model_version = get_model_version(INCLUSION_MODEL_RELPATH)
    exclusion_threshold = get_exclusion_threshold(LOCAL_THRESHOLD_RELPATH)
    n_results = get_n_results(pubmed_credentials, SEARCH_QUERY, START_DATE, END_DATE)
    previous_pmids = get_previous_pmids(LOCAL_LOG_RELPATH)
    pmids = build_pmid_list(n_results, previous_pmids, pubmed_credentials, SEARCH_QUERY, ORIGINAL_START_DATE, START_DATE, END_DATE)
    ds = retrieve_pubmed_data(pmids, pubmed_credentials)
    verify_pubmed_retrieval(ds)
    filtered_ds = build_text_and_filter_dataset(ds, ABSTRACT_SECTIONS_TO_EXCLUDE)
    df = make_inclusion_predictions(filtered_ds, exclusion_threshold, model_version, INCLUSION_MODEL_RELPATH)
    rows_to_append = convert_df_to_rows(df)
    update_local_data(pmids, START_DATE, END_DATE, exclusion_threshold, LOCAL_LOG_RELPATH)
    update_google_sheet(google_sheet, DATA_SHEET_NAME, LOG_SHEET_NAME, rows_to_append, START_DATE, END_DATE, pmids, exclusion_threshold)
    print('DONE !')
