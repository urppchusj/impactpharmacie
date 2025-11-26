import gspread
import json
import math
import os
import pandas as pd
import requests
import time
import logging

from bs4 import BeautifulSoup
from collections import defaultdict
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# CONSTANTS AND CONFIGURATION
FILEPATH = '.'
GOOGLE_SPREADSHEET_ID = 'REAL'
DATA_SHEET_NAME = 'data'
LOG_SHEET_NAME = 'extraction_log'
LOCAL_LOG_RELPATH = '/data/second_gen/extraction_log.csv'
INCLUSION_MODELS_TO_USE = {
    'inclusion_biobert': {
        'relpath': '/models/production_models/inclusion_biobert',
        'threshold_relpath': '/data/second_gen/thresholds_biobert.csv',
        'model_source': 'dmis-lab/biobert-base-cased-v1.2'
    },
    'inclusion_biomedbert': {
        'relpath': '/models/production_models/inclusion_biomedbert',
        'threshold_relpath': '/data/second_gen/thresholds_biomedbert.csv',
        'model_source': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
    },
}
BASELINE_INCLUSION_MODEL_TO_USE = 'inclusion_biobert'
ORIGINAL_START_DATE = '2021/11/07'
START_DATE = '2025/11/16'
END_DATE = '2025/11/22'
SEARCH_QUERY = 'pharmacists[All Fields] OR pharmacist[All Fields] OR pharmacy[title]'
ABSTRACT_SECTIONS_TO_EXCLUDE = ['DISCLAIMER']
PUBMED_BATCH_SIZE = 100000
SLEEP_BETWEEN_REQUESTS = 0.35
SLEEP_BETWEEN_PMID_FETCH = 0.4

# Setup logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def get_google_sheet(google_spreadsheet_id, data_sheet_name):
    """Authenticate and return the Google Sheet object."""
    credentials_filepath = os.path.join(FILEPATH, 'credentials/credentials.json')
    authorized_user_filepath = os.path.join(FILEPATH, 'credentials/authorized_user.json')
    try:
        # Try to authenticate and open the sheet
        gc = gspread.oauth(
            credentials_filename=credentials_filepath,
            authorized_user_filename=authorized_user_filepath
        )
        sht = gc.open_by_key(google_spreadsheet_id)
        sht.worksheet(data_sheet_name)
    except Exception:
        # If authentication fails, remove the authorized user file and retry
        if os.path.exists(authorized_user_filepath):
            os.remove(authorized_user_filepath)
        gc = gspread.oauth(
            credentials_filename=credentials_filepath,
            authorized_user_filename=authorized_user_filepath
        )
        sht = gc.open_by_key(google_spreadsheet_id)
        sht.worksheet(data_sheet_name)
    return sht

def get_model_version(inclusion_model_relpath):
    """Return the latest model version as an int."""
    # Extract version numbers from directory names (e.g., v1, v2, ...)
    versions = [int(x[1:]) for x in os.listdir(FILEPATH + inclusion_model_relpath) if x.startswith('v') and x[1:].isdigit()]
    version = max(versions)
    logging.info(f'Using inclusion model version: {version}')
    return version

def get_exclusion_threshold(local_threshold_relpath):
    """Return the latest computed threshold as float."""
    threshold_df = pd.read_csv(FILEPATH + local_threshold_relpath)
    threshold = float(threshold_df.iloc[-1]['computed_threshold'])
    logging.info(f'Using exclusion threshold: {threshold:.5f}')
    return threshold

def get_n_results(pubmed_credentials, search_query, start_date, end_date):
    """Return the number of PubMed results for the query."""
    params = {
        'db': 'pubmed', 'term': search_query, 'datetype': 'edat',
        'mindate': start_date, 'maxdate': end_date,
        'tool': pubmed_credentials['pubmed_tool_name'],
        'email': pubmed_credentials['pubmed_tool_email']
    }
    try:
        # Query PubMed for the count of results
        r = requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi', params=params, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, 'xml')
        n_results = int(soup.find('Count').text)
        logging.info(f'Results for current extraction: {n_results}')
        return n_results
    except Exception as e:
        logging.error(f"Error fetching PubMed results: {e}")
        return 0

def get_previous_pmids(local_log_relpath):
    """Return a set of previously extracted PMIDs."""
    extraction_log_df = pd.read_csv(FILEPATH + local_log_relpath, index_col=0).fillna('')
    pmid_lists = extraction_log_df['pmids'].map(lambda x: x.split(', '))
    # Flatten the list of lists into a set for fast lookup
    return set(pmid for sublist in pmid_lists for pmid in sublist)

def fetch_pmids_batch(pubmed_credentials, search_query, mindate, maxdate, retstart, retmax):
    """Fetch a batch of PMIDs from PubMed."""
    params = {
        'db': 'pubmed', 'term': search_query, 'datetype': 'edat',
        'mindate': mindate, 'maxdate': maxdate,
        'retstart': retstart, 'retmax': retmax,
        'tool': pubmed_credentials['pubmed_tool_name'],
        'email': pubmed_credentials['pubmed_tool_email']
    }
    try:
        # Request a batch of PMIDs from PubMed
        r = requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi', params=params, timeout=10)
        r.raise_for_status()
        results = BeautifulSoup(r.content, 'xml')
        return [id_tag.text for id_tag in results.find_all('Id')]
    except Exception as e:
        logging.error(f"Error fetching PMIDs batch: {e}")
        return []

def build_pmid_list(n_results, previous_pmids, pubmed_credentials, search_query, original_start_date, start_date, end_date):
    """Build the list of PMIDs to fetch, including missed ones."""
    def get_all_pmids(mindate, maxdate):
        # Fetch all PMIDs in batches for a given date range
        n_calls = math.ceil(n_results / PUBMED_BATCH_SIZE)
        pmids = []
        for i in tqdm(range(n_calls), desc=f"Fetching PMIDs {mindate} to {maxdate}"):
            pmids.extend(fetch_pmids_batch(pubmed_credentials, search_query, mindate, maxdate, i * PUBMED_BATCH_SIZE, PUBMED_BATCH_SIZE))
            time.sleep(SLEEP_BETWEEN_REQUESTS)
        return set(pmids)

    # Get current and potentially missed PMIDs
    current_pmids = get_all_pmids(start_date, end_date)
    potentially_missed_pmids = get_all_pmids(original_start_date, end_date)
    missed_pmids = potentially_missed_pmids - previous_pmids - current_pmids

    logging.info(f'Number of previously missed PMIDs: {len(missed_pmids)}')
    # Combine current and missed PMIDs, excluding already processed ones
    pmids = list((current_pmids - previous_pmids - missed_pmids) | missed_pmids)
    logging.info(f'Number of PMIDs to retrieve: {len(pmids)}')
    return pmids

def retrieve_pubmed_data(pmids, pubmed_credentials):
    """Retrieve PubMed XML data for each PMID."""
    dataset = {}
    for pmid in tqdm(pmids, desc="Fetching PubMed data"):
        params = {
            'db': 'pubmed', 'id': pmid, 'retmode': 'xml',
            'tool': pubmed_credentials['pubmed_tool_name'],
            'email': pubmed_credentials['pubmed_tool_email']
        }
        try:
            # Fetch XML data for each PMID
            r = requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', params=params, timeout=10)
            r.raise_for_status()
            dataset[pmid] = {'pmdata': str(BeautifulSoup(r.content, 'xml'))}
        except Exception as e:
            dataset[pmid] = {'pmdata': None}
            logging.error(f'Error fetching pmid {pmid}: {e}')
        time.sleep(SLEEP_BETWEEN_PMID_FETCH)
    return dataset

def verify_pubmed_retrieval(ds):
    """Check for missing PubMed data."""
    n_none = sum(1 for v in ds.values() if v['pmdata'] is None)
    if n_none > 0:
        error_pmids = [pmid for pmid, v in ds.items() if v['pmdata'] is None]
        logging.error(f'ERROR in PubMed retrieval for PMIDs: {error_pmids}')
        raise RuntimeError("PubMed retrieval errors detected.")

def build_text_and_filter_dataset(ds, abstract_sections_to_exclude):
    """Extract title and abstract text, filter out empty entries."""
    filtered_dataset = {}
    for pmid, data in ds.items():
        if data['pmdata'] is None:
            continue
        element_pmdata = BeautifulSoup(data['pmdata'], 'xml')
        # Extract title
        title = (element_pmdata.find('ArticleTitle').get_text() if element_pmdata.find('ArticleTitle') else '')
        # Extract abstract, filter out excluded sections
        abstract = element_pmdata.find('Abstract')
        if abstract:
            labels_texts = [
                (e.get('Label', ''), e.get_text())
                for e in abstract.find_all('AbstractText')
            ]
            labelsandtext = ' '.join(
                f"{l} {t}" for l, t in labels_texts if l not in abstract_sections_to_exclude
            )
        else:
            labelsandtext = ''
        text = f"{title} {labelsandtext}".strip()
        if text:
            filtered_dataset[pmid] = {'text': text}
    logging.info(f'Number of elements in filtered dataset: {len(filtered_dataset)}')
    return filtered_dataset

def make_inclusion_predictions(df, exclusion_threshold, model_name, model_source, model_version, inclusion_model_relpath):
    """Run inclusion model and add predictions to DataFrame using efficient batch processing."""
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(
        FILEPATH + inclusion_model_relpath + f'/v{model_version}', local_files_only=True
    )
    # Use pipeline with batch support
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        device=0
    )
    texts = df['text'].tolist()
    results = pipe(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        batch_size=32,
        top_k=None
    )
    # The output is a list of lists of dicts (one list per input, one dict per class)
    # We assume the positive class is at index 1 (as in your original code)
    scores = [r[1]['score'] for r in results]
    df[f'{model_name}_suggestion'] = ['Exclude' if x < exclusion_threshold else 'Review' for x in scores]
    df[f'{model_name}_model_version'] = model_version
    if model_name == BASELINE_INCLUSION_MODEL_TO_USE:
        df['rating1'] = ''
        df['rating2'] = ''
        df['rating_consensus'] = [0 if s == 'Exclude' else '' for s in df[f'{model_name}_suggestion']]
        df['consensus_reason'] = [
            f'Excluded by ML model {model_name}' if s == 'Exclude' else '' for s in df[f'{model_name}_suggestion']
        ]
    logging.info(f'For inclusion model {model_name}: Excluded={df[f"{model_name}_suggestion"].value_counts().get("Exclude",0)}, Review={df[f"{model_name}_suggestion"].value_counts().get("Review",0)}')
    return df

def convert_df_to_rows(df):
    """Convert DataFrame to list of rows for Google Sheets."""
    cols = [
        'PMID', 'text', 'inclusion_biobert_suggestion', 'inclusion_biobert_model_version',
        'inclusion_biomedbert_suggestion', 'inclusion_biomedbert_model_version',
        'rating1', 'rating2', 'rating_consensus', 'consensus_reason'
    ]
    # Convert DataFrame to list of lists for Google Sheets API
    return df.reset_index().rename({'index': 'PMID'}, axis=1)[cols].values.tolist()

def update_local_data(pmids, start_date, end_date, exclusion_threshold, inclusion_model_used, local_log_relpath):
    """Append new extraction info to local log."""
    df_to_append = pd.DataFrame([{
        'date_begin': start_date, 'date_end': end_date, 'n_results': len(pmids),
        'pmids': ', '.join(pmids), 'exclusion_threshold': exclusion_threshold,
        'baseline_inclusion_model_name': inclusion_model_used
    }])
    extraction_log_df = pd.read_csv(FILEPATH + local_log_relpath, index_col=0).fillna('')
    updated_extraction_log = pd.concat([extraction_log_df, df_to_append], ignore_index=True)
    updated_extraction_log.to_csv(FILEPATH + local_log_relpath)

def update_google_sheet(sht, data_sheet_name, log_sheet_name, rows_to_append, start_date, end_date, pmids, exclusion_threshold, inclusion_model_used):
    """Update Google Sheet with new data and log."""
    data_sheet = sht.worksheet(data_sheet_name)
    # Clear and update data contents
    data_sheet.batch_clear(['data_contents'])
    data_sheet.append_rows(rows_to_append)
    # Hide columns as per requirements
    data_sheet.hide_columns(4, 6)
    data_sheet.hide_columns(11, 13)
    data_sheet.hide_columns(14, 17)
    data_sheet.hide_columns(18, 19)
    # Log extraction in log sheet
    log_sheet = sht.worksheet(log_sheet_name)
    log_sheet.append_row([start_date, end_date, len(pmids), ', '.join(pmids), exclusion_threshold, inclusion_model_used])

def main():
    """Main workflow for updating data and predictions."""
    # Load spreadsheet and PubMed credentials
    with open(FILEPATH + '/credentials/spreadsheet_ids.json', mode='r') as file:
        spreadsheet_ids = json.load(file)
    google_spreadsheet_id = spreadsheet_ids['real_google_spreadsheet_id'] if GOOGLE_SPREADSHEET_ID == 'REAL' else spreadsheet_ids['test_google_spreadsheet_id']
    with open(FILEPATH + '/credentials/pubmed_credentials.json', mode='r') as file:
        pubmed_credentials = json.load(file)
    # Connect to Google Sheet
    google_sheet = get_google_sheet(google_spreadsheet_id, DATA_SHEET_NAME)
    # Get number of results and previous PMIDs
    n_results = get_n_results(pubmed_credentials, SEARCH_QUERY, START_DATE, END_DATE)
    previous_pmids = get_previous_pmids(LOCAL_LOG_RELPATH)
    # Build list of PMIDs to fetch
    pmids = build_pmid_list(n_results, previous_pmids, pubmed_credentials, SEARCH_QUERY, ORIGINAL_START_DATE, START_DATE, END_DATE)
    # Retrieve PubMed data and verify
    ds = retrieve_pubmed_data(pmids, pubmed_credentials)
    verify_pubmed_retrieval(ds)
    # Build and filter dataset for prediction
    filtered_ds = build_text_and_filter_dataset(ds, ABSTRACT_SECTIONS_TO_EXCLUDE)
    df = pd.DataFrame.from_dict(filtered_ds, orient='index', columns=['text'])
    # Run predictions for each inclusion model
    for inclusion_model_name, inclusion_model_parameters in INCLUSION_MODELS_TO_USE.items():
        logging.info(f'Preparing inclusion model for prediction: {inclusion_model_name}')
        inclusion_model_version = get_model_version(inclusion_model_parameters['relpath'])
        inclusion_model_exclusion_threshold = get_exclusion_threshold(inclusion_model_parameters['threshold_relpath'])
        if inclusion_model_name == BASELINE_INCLUSION_MODEL_TO_USE:
            exclusion_threshold = inclusion_model_exclusion_threshold
            inclusion_model_used = inclusion_model_name
        df = make_inclusion_predictions(df, inclusion_model_exclusion_threshold, inclusion_model_name, inclusion_model_parameters['model_source'], inclusion_model_version, inclusion_model_parameters['relpath'])
    # Convert results and update logs and Google Sheet
    rows_to_append = convert_df_to_rows(df)
    update_local_data(pmids, START_DATE, END_DATE, exclusion_threshold, inclusion_model_used, LOCAL_LOG_RELPATH)
    update_google_sheet(google_sheet, DATA_SHEET_NAME, LOG_SHEET_NAME, rows_to_append, START_DATE, END_DATE, pmids, exclusion_threshold, inclusion_model_used)
    logging.info('DONE!')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f'Fatal error: {e}')
