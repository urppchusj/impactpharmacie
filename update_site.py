import gspread
import json
import pandas as pd
import os
import pickle
import requests
import time

from bs4 import BeautifulSoup
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import cohen_kappa_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm

# GLOBAL VARS
FILEPATH = '.'
GOOGLE_SPREADSHEET_ID = 'REAL' # 'REAL' OR 'TEST'
DATA_SHEET_NAME = 'data' # NAME OF DATA SHEET IN SPREADSHEET
LOG_SHEET_NAME = 'extraction_log' # NAME OF LOG SHEET IN SPREADSHEET
PREDICTION_SHEET_NAME = 'machine_learning_predictions' # NAME OF PREDICTIONS SHEET IN SPREADSHEET
LOCAL_DATA_RELPATH = '/data/second_gen/ratings.csv'
LOCAL_LOG_RELPATH = '/data/second_gen/extraction_log.csv'
LOCAL_PREDICTIONS_RELPATH = '/data/second_gen/predictions.csv'
ORIGINAL_START_DATE = '2021/11/07' # FORMAT 'YYYY/MM/DD'
START_DATE = '2022/05/08' # FORMAT 'YYYY/MM/DD'
END_DATE = '2022/05/14' # FORMAT 'YYYY/MM/DD'
SEARCH_QUERY = 'pharmacists[All Fields] OR pharmacist[All Fields] OR pharmacy[title]' # PUBMED QUERY STRING
MAX_PUBMED_TRIES = 10 # NUMBER OF MAXIMUM PUBMED QUERY TRIES BEFORE GIVING UP
ABSTRACT_SECTIONS_TO_EXCLUDE = ['DISCLAIMER'] # List of abstract labels that will be excluded from data 
TAGS_TO_USE = {'design':{'column':'design_pred', 'version':1}, 'field':{'column':'field_ground_truth', 'version':'0.1'}, 'setting':{'column':'setting_ground_truth','version':'0.1'}} # Dict with model strings as keys, values are dicts with column to use in dataframe as the first key and value and model version as second key and value
DESIGN_LABEL_TRANSLATIONS = {'Study':'Étude', 'Systematic review or meta-analysis':'Revue systématique ou méta-analyse'}
FIELDS_LABELS_TRANSLATIONS = {'Anticoagulation':'Anticoagulation', 'Cardiology':'Cardiologie', 'Critical care':'Soins critiques', 'Emergency medicine':'Urgence', 'Geriatric':'Gériatrie', 'Infectious diseases':'Infectiologie', 'Oncology':'Oncologie', 'Palliative care':'Soins palliatifs', 'Maternal / pediatric / neonatal':'Soins mère-enfant / pédiatrie / néonatologie', 'Psychiatric':'Psychiatrie', 'Solid organ transplantation':'Transplantation', 'Other':'Autre'}
SETTING_LABELS_TRANSLATIONS = {'Ambulatory':'Ambulatoire', 'Community':'Communautaire', 'Inpatient':'Établissement', 'Nursing home':'Soins de longue durée', 'Other':'Autre'}
TRANSLATION_DICT = {**DESIGN_LABEL_TRANSLATIONS, **FIELDS_LABELS_TRANSLATIONS, **SETTING_LABELS_TRANSLATIONS}
MONTH_NAMES_ENG = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
MONTH_NAMES_FR = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']

# FUNCTIONS

def get_google_sheets(google_spreadsheet_id, data_sheet_name, prediction_sheet_name):
    credentials_filepath = FILEPATH + '/credentials/credentials.json'
    authorized_user_filepath = FILEPATH + '/credentials/authorized_user.json'
    gc = gspread.oauth(
        credentials_filename = credentials_filepath,
        authorized_user_filename = authorized_user_filepath
    )
    try:
        sht = gc.open_by_key(google_spreadsheet_id)
        ratings_sheet = sht.worksheet(data_sheet_name)
    except:
        if os.path.exists(authorized_user_filepath):
            os.remove(authorized_user_filepath)
        sht = gc.open_by_key(google_spreadsheet_id)
        ratings_sheet = sht.worksheet(data_sheet_name)
    predictions_sheet = sht.worksheet(prediction_sheet_name)
    return ratings_sheet, predictions_sheet

def verify_and_validate_data(local_log_relpath, local_data_relpath, ratings_sheet, start_date, end_date):

    extraction_log_df = pd.read_csv(FILEPATH + local_log_relpath, index_col=0).fillna('')
    extraction_log_df['pmids'] = extraction_log_df['pmids'].map(lambda x:x.split(', '))

    old_ratings_df = pd.read_csv(FILEPATH + local_data_relpath, index_col=0, dtype=str).fillna('')
    new_ratings_df = pd.DataFrame(ratings_sheet.get_all_records())
    ratings_df = pd.concat([old_ratings_df, new_ratings_df[1:]], ignore_index=True)[1:]
    ratings_df['rating_final'] = ratings_df.apply(lambda x: x['rating_consensus'] if x['rating_consensus'] != '' else x['rating1'] if ((x['rating1'] == x['rating2']) and (x['consensus_reason'] == '') and (x['rating1'] != '')) else 'error', axis=1)
    ratings_df['design_final'] = ratings_df['design_ground_truth']
    ratings_df['field_final'] = ratings_df.apply(lambda x: x['field_ground_truth_consensus'] if x['field_ground_truth_consensus'] != '' else x['field_ground_truth_1'] if ((x['field_ground_truth_1'] == x['field_ground_truth_2']) and (x['field_ground_truth_consensus_reason'] == '') and (x['field_ground_truth_1'] != '')) else 'error', axis=1)
    ratings_df['setting_final'] = ratings_df.apply(lambda x: x['setting_ground_truth_consensus'] if x['setting_ground_truth_consensus'] != '' else x['setting_ground_truth_1'] if ((x['setting_ground_truth_1'] == x['setting_ground_truth_2']) and (x['setting_ground_truth_consensus_reason'] == '') and (x['setting_ground_truth_1'] != '')) else 'error', axis=1)

    if 'error' in ratings_df['rating_final'].unique().tolist():
        print('ERROR in ratings, please verify data')
        quit()
    else:
        print('No error in ratings')

    if 'error' in ratings_df['field_final'].unique().tolist():
        print('ERROR in field labels, please verify data')
        quit()
    else:
        print('No error in field labels')

    if 'error' in ratings_df['setting_final'].unique().tolist():
        print('ERROR in setting labels, please verify data')
        quit()
    else:
        print('No error in setting labels')

    if start_date in extraction_log_df['date_begin'].tolist():
        selected_extraction_df = extraction_log_df.loc[extraction_log_df['date_begin'] == start_date].copy().reset_index()
        if len(selected_extraction_df) != 1:
            print('ERROR, no extraction or more than one extraction with selected start date, verify data')
            quit()
        else:
            if end_date not in selected_extraction_df['date_end'].tolist():
                print('ERROR, selected end date does not match selected start date given currently available extractions, verify data')
                quit()
            else:
                pass
    else:
        print('ERROR, start date is not in extraction logs')
        quit()

    ratings_df['included_in_selected_extraction'] = ratings_df.apply(lambda x:str(x['PMID']) in selected_extraction_df.at[0, 'pmids'], axis=1)

    included_df = ratings_df.loc[(ratings_df['rating_final'] == 1) & (ratings_df['included_in_selected_extraction'] == True)].copy()

    current_extraction_df = ratings_df.loc[ratings_df['included_in_selected_extraction'] == 1].copy()

    pmids = included_df['PMID'].tolist()

    return(extraction_log_df, selected_extraction_df, ratings_df, included_df, current_extraction_df, pmids)

def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
    tokenizer.model_max_length = 512
    return(tokenizer)

def prepare_predictions_dict(df):
    predictions = defaultdict()
    for row in df.iterrows():
        predictions[row[1]['PMID']] = defaultdict() 
    return(predictions)

def make_predictions(tokenizer, filepath, model_string, model_version, df, predictions_dict):
    model = AutoModelForSequenceClassification.from_pretrained(filepath + '/models/production_models/{}_biobert/v{}/'.format(model_string, model_version), local_files_only=True).cuda()

    with open(filepath + '/models/production_models/{}_biobert/v{}/{}_labels.pkl'.format(model_string, model_version, model_string), mode='rb') as file:
        labels = pickle.load(file)

    current_labelsids = model.config.id2label
    new_id2label = {o[0]:n for o, n in zip(current_labelsids.items(), labels)}
    new_label2id = {v:k for k, v in new_id2label.items()}
    model.config.label2id = new_label2id
    model.config.id2label = new_id2label

    pipe = pipeline('text-classification', model=model, tokenizer=tokenizer, device=0)

    for row in df.iterrows():
        prediction = pipe(row[1]['text'], truncation=True)[0]['label']
        predictions_dict[row[1]['PMID']]['{}_pred'.format(model_string)] = prediction
        predictions_dict[row[1]['PMID']]['{}_model_version'.format(model_string)] = model_version
        predictions_dict[row[1]['PMID']]['{}_ground_truth'.format(model_string)] = row[1]['{}_final'.format(model_string)]

    return(predictions_dict)

def convert_prediction_dict_to_df(prediction_dict):
    pred_df = pd.DataFrame.from_dict(prediction_dict, orient='index')
    return (pred_df)

def update_prediction_local_data(local_predictions_relpath, pred_df):
    pred_df = pred_df.reset_index().rename({'index':'PMID'},axis=1)
    old_pred_df = pd.read_csv(FILEPATH + local_predictions_relpath, index_col=0).fillna('')
    updated_pred_df = pd.concat([old_pred_df, pred_df], ignore_index=True)
    updated_pred_df.to_csv(FILEPATH + local_predictions_relpath)

def update_prediction_google_sheet(prediction_sheet, pred_df):
    rows_to_append = pred_df.reset_index().rename({'index':'PMID'},axis='columns').values.tolist()
    prediction_sheet.append_rows(rows_to_append)

def update_ratings_local_data(local_data_relpath, ratings_sheet):
    old_ratings_df = pd.read_csv(FILEPATH + local_data_relpath, index_col=0, dtype=str).fillna('')
    new_ratings_df = pd.DataFrame(ratings_sheet.get_all_records())
    ratings_df = pd.concat([old_ratings_df, new_ratings_df[1:]], ignore_index=True)
    ratings_df.to_csv(FILEPATH + local_data_relpath)

def make_prediction_tags(pred_df, tag_columns_to_use, translation_dict):
    predictions_tags_keys = pred_df.index.values
    prediction_tags_values = pred_df[tag_columns_to_use].values.tolist()
    tag_lists = []
    for tag_list in prediction_tags_values:
        flattened_tag_list = []
        for string in tag_list:
            for el in string.split(', '):
                flattened_tag_list.append(el)
        tag_lists.append(flattened_tag_list)
    prediction_tags_eng = {k:[el for el in v if el not in ['', 'Other'] ]for k,v in zip(predictions_tags_keys, tag_lists)}
    prediction_tags_fr = {k:[translation_dict[el] for el in v if el not in ['', 'Other'] ]for k,v in zip(predictions_tags_keys, tag_lists)}
    prediction_tags_all = {k:[*ien, *prediction_tags_fr[k]] for k, ien in prediction_tags_eng.items()}
    return(prediction_tags_eng, prediction_tags_fr, prediction_tags_all)

def retrieve_pubmed_data(pmids, pubmed_credentials):
    dataset = defaultdict()
    for pmid in tqdm(pmids):
        dataset[pmid] = defaultdict()
        pmdata = get_pubmed_single_pmid(pmid, pubmed_credentials)
        dataset[pmid]['pmdata'] = pmdata
    time.sleep(0.35)
    return(dataset)

def get_pubmed_single_pmid(pmid, pubmed_credentials, ntries=0):
    params = {'db':'pubmed', 'id':pmid, 'retmode':'xml', 'tool':pubmed_credentials['pubmed_tool_name'], 'email':pubmed_credentials['pubmed_tool_email']}
    r = requests.get(url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', params=params)
    ntries += 1
    try:
        pmdata = str(BeautifulSoup(r.content, 'xml'))
    except:
        print('ERROR on call for pmid {}, tried {} times, will retry for max {} times'.format(pmid, ntries, MAX_PUBMED_TRIES))
        if ntries <= MAX_PUBMED_TRIES:
            time.sleep(0.35)
            pmdata = get_pubmed_single_pmid(pmid, pubmed_credentials, ntries=ntries)
        else:
            print('ERROR, PubMed query error after {} tries, verify settings'.format(ntries))
            quit()
    return pmdata
    
def verify_pubmed_retrieval(ds):
    print('Length of dataset: {}'.format(len(ds)))
    n_pubmed = 0
    n_none = 0
    error_pmids = []
    for pmid, data in ds.items():
        if data['pmdata'] != None:
            n_pubmed += 1
        else:
            n_none += 1
    print('Number of successfully retrieved PubMed sets: {}'.format(n_pubmed))
    print('Number of errors: {}'.format(n_none))
    if n_none > 0:
        print('ERROR in PubMed retrieval for following PMIDs, verify data: {}'.format(error_pmids))
        quit()

def rebuild_dataset(ds, prediction_tags_fr, prediction_tags_eng, prediction_tags_all, pubmed_credentials):
    for pmid, data in ds.items():
        title, labels, texts, authors, journal, doi, pubdate = process_single_pmid_data(pmid, data, pubmed_credentials)
        ds[pmid]['title'] = title
        ds[pmid]['labels'] = labels
        ds[pmid]['texts'] = texts
        ds[pmid]['authors'] = authors
        ds[pmid]['journal'] = journal
        ds[pmid]['doi'] = doi
        ds[pmid]['pubdate'] = pubdate
        ds[pmid]['machine_learning_tags_fr'] = prediction_tags_fr[pmid]
        ds[pmid]['machine_learning_tags_eng'] = prediction_tags_eng[pmid]
        ds[pmid]['machine_learning_tags_all'] = prediction_tags_all[pmid]
    return(ds)

def process_single_pmid_data(pmid, data, pubmed_credentials, ntries=0):
    title = ''
    labels = []
    texts = []
    if data['pmdata'] == None:
        ds[pmid]['texts'] = []
    else:
        ntries += 1
        try:
            element_pmdata = BeautifulSoup(data['pmdata'], 'xml')
            try:
                t = element_pmdata.find_all('ArticleTitle')
                title = [e.get_text() for e in t][0]
                if title == None:
                    title = ''
            except:
                title = ''
            if element_pmdata.find('Abstract') == None:
                pass
            else:
                labels = [e['Label'] if 'Label' in e.attrs.keys() else '' for e in element_pmdata.find('Abstract').find_all('AbstractText')]
                texts = [e.get_text() for e in element_pmdata.find('Abstract').find_all('AbstractText')]
            authors = [' '.join([f.get_text(), l.get_text()]) for f, l in zip(element_pmdata.find_all('ForeName'), element_pmdata.find_all('LastName'))]
            journal = element_pmdata.find('MedlineTA').get_text()
            doi_list = [e.get_text() for e in element_pmdata.find_all('ArticleId') if 'doi' in e.attrs.values()]
            if len(doi_list) > 0:
                doi = doi_list[0]
            else:
                doi = ''
            pubdate = ' '.join([e.get_text() for e in element_pmdata.find('PubDate').children if e.name in ['Year', 'Month', 'Day']])
        except Exception as e:
            print('ERROR processing data for pmid {}, error: {} . Tried {} times, will retry for max {} times'.format(pmid, e.message, ntries, MAX_PUBMED_TRIES))
            if ntries <= MAX_PUBMED_TRIES:
                time.sleep(0.35)
                data['pmdata'] = get_pubmed_single_pmid(pmid, pubmed_credentials, ntries=ntries)
                title, labels, texts, authors, journal, doi, pubdate = process_single_pmid_data(pmid, data, pubmed_credentials)
            else:
                print('ERROR processing data for pmid {} after {} tries, verify settings'.format(ntries))
                quit()
    return title, labels, texts, authors, journal, doi, pubdate

def verify_and_filter_dataset(ds):
    print('Number of elements in dataset: {}'.format(len(ds)))

    title_available = 0
    for _,data in ds.items():
        if data['title'] == '':
            continue
        else:
            title_available += 1
    print('Number of elements with titles: {}'.format(title_available))

    labels_available = 0
    for _,data in ds.items():
        if len(data['labels']) == 0:
            continue
        else:
            labels_available += 1
    print('Number of elements with labels: {}'.format(labels_available))

    text_data_available = 0
    for _,data in ds.items():
        if len(data['texts']) == 0:
            continue
        else:
            text_data_available += 1
    print('Number of elements with text data: {}'.format(text_data_available))

    filtered_dataset = defaultdict()
    for key,data in ds.items():
        if len(data['texts']) == 0 or data['title'] == '':
            continue
        else:
            filtered_dataset[key] = data
    print('Number of elements in filtered dataset: {}'.format(len(filtered_dataset)))
    return (filtered_dataset)

def publications_posts(ds, post_url, header):
    categories = ['Publications']

    post_template = '<!-- wp:paragraph {{"fontSize":"small"}} --><p class="has-small-font-size">PMID: <a rel="noreferrer noopener" href="https://pubmed.ncbi.nlm.nih.gov/{}/" target="_blank">{}</a>{}<br><em>{}</em>, <em>{}</em></p><!-- /wp:paragraph -->{}<!-- wp:paragraph {{"fontSize":"small"}} --><p class="has-small-font-size">{}</p><!-- /wp:paragraph -->'

    for pmid, data in ds.items():
        post_content = post_template.format(
            pmid,
            pmid,
            '     doi: <a rel="noreferrer noopener" href="https://dx.doi.org/{}" target="_blank">{}</a>'.format(data['doi'], data['doi']) if data['doi'] != '' else '',
            data['journal'],
            data['pubdate'],
            ' '.join([' '.join(['<!-- wp:paragraph --><p><strong>'+l+'</strong>',t+'</p><!-- /wp:paragraph -->']) for l,t in zip(data['labels'], data['texts']) if l != 'DISCLAIMER']),
            ', '.join(data['authors'])+'.'
            )
        title=data['title']
        post_tags_list = data['machine_learning_tags_all']
        response = requests.post(post_url, headers=header, data={'slug': pmid, 'title':title, 'content':post_content, 'categories':categories, 'tags':','.join(post_tags_list)})
        print('Publication update post response for PMID {}: {}'.format(pmid, response))

def french_update_post(month_names, start_date, end_date, selected_extraction_df, extraction_log_df, current_extraction_df, ratings_df, ds, post_url, header):
    categories_update = ['Mise à jour des données']
    update_post_template = '<!-- wp:paragraph --><p>Cette mise à jour couvre la période du {} au {}.</p><!-- /wp:paragraph --><!-- wp:paragraph --><p>{} publications ont été identifiées. {} publications ont été retenues pour un taux d\'inclusion de {:.1f}%. Le kappa entre les réviseurs était de {:.3f}.</p><!-- /wp:paragraph --><!-- wp:paragraph --><p>Les publications suivantes ont été retenues dans cette mise à jour:</p><!-- /wp:paragraph --><!-- wp:list --><ul>{}</ul><!-- /wp:list --><!-- wp:paragraph --><p>Depuis novembre 2021, {} publications ont été évaluées dont {} ont été retenues, pour un taux d\'inclusion de {:.1f}%. Le kappa entre les réviseurs pour toutes les publications évaluées est de {:.3f}.</p><!-- /wp:paragraph -->'

    update_post_content = update_post_template.format(
        start_date, 
        end_date, 
        selected_extraction_df.at[0, 'n_results'], 
        len(ds), 
        (len(ds) / selected_extraction_df.at[0, 'n_results']) *100, 
        cohen_kappa_score(current_extraction_df['rating1'].astype(int), current_extraction_df['rating2'].astype(int)), 
        ''.join(['<li><a href="https://impactpharmacie.net/{}">{}</a> - {}</li>'.format(
            pmid, 
            data['title'],
            ' - '.join([tag for tag in data['machine_learning_tags_fr']]))
            for pmid, data in ds.items()]),
        len(ratings_df),
        ratings_df['rating_final'].astype(int).sum(), 
        (ratings_df['rating_final'].astype(int).sum() / len(ratings_df)) *100, 
        cohen_kappa_score(ratings_df['rating1'].astype(int), ratings_df['rating2'].astype(int))
        )
    update_post_title='Mise à jour du {} {} {}'.format(time.localtime()[2], month_names[time.localtime()[1]-1], time.localtime()[0])
    response = requests.post(post_url, headers=header, data={'title':update_post_title, 'content':update_post_content, 'categories':categories_update})
    print('French update post response: {}'.format(response))

def english_update_post(month_names, start_date, end_date, selected_extraction_df, extraction_log_df, current_extraction_df, ratings_df, ds, post_url, header):
    categories_update = ['Data update']
    update_post_template = '<!-- wp:paragraph --><p>This update covers publications from {} to {}.</p><!-- /wp:paragraph --><!-- wp:paragraph --><p>{} publications were identified. {} publications were selected, for an inclusion rate of {:.1f}%. Inter-rater kappa was {:.3f}.</p><!-- /wp:paragraph --><!-- wp:paragraph --><p>The following publications were included in this update:</p><!-- /wp:paragraph --><!-- wp:list --><ul>{}</ul><!-- /wp:list --><!-- wp:paragraph --><p>Since November 2021, {} publications were evaluated. Among these, {} were selected, for an inclusion rate of {:.1f}%. Inter-rater kappa for all evaluated publications is {:.3f}.</p><!-- /wp:paragraph -->'

    update_post_content = update_post_template.format(
        start_date, 
        end_date, 
        selected_extraction_df.at[0, 'n_results'], 
        len(ds), 
        (len(ds) / selected_extraction_df.at[0, 'n_results']) *100, 
        cohen_kappa_score(current_extraction_df['rating1'].astype(int), current_extraction_df['rating2'].astype(int)), 
        ''.join(['<li><a href="https://impactpharmacie.net/{}">{}</a> - {}</li>'.format(
            pmid, 
            data['title'],
            ' - '.join([tag for tag in data['machine_learning_tags_eng']]) 
            ) for pmid, data in ds.items()]),
        len(ratings_df),
        ratings_df['rating_final'].astype(int).sum(), 
        (ratings_df['rating_final'].astype(int).sum() / len(ratings_df)) *100, 
        cohen_kappa_score(ratings_df['rating1'].astype(int), ratings_df['rating2'].astype(int))
        )
    update_post_title='Data update for {} {}, {}'.format(month_names[time.localtime()[1]-1], time.localtime()[2], time.localtime()[0])
    response = requests.post(post_url, headers=header, data={'title':update_post_title, 'content':update_post_content, 'categories':categories_update})
    print('English update post response: {}'.format(response))

def make_newsletter_post(start_date, end_date, selected_extraction_df, extraction_log_df, current_extraction_df, ds, post_url, header):

    categories_briefing = ['Impact Briefing']

    briefing_update_text_template = '<!-- wp:paragraph --><p><a href="#{}summary">English</a></p><!-- /wp:paragraph --><!-- wp:spacer {{"height":40}} --><div style="height:40px" aria-hidden="true" class="wp-block-spacer"></div><!-- /wp:spacer --><p><a name="{}resume"></a></p><!-- wp:heading --><h2 id="{}resume">Résumé</h2><!-- /wp:heading --><!-- wp:paragraph --><p>Ce Impact Briefing couvre la période du {} au {}.</p><!-- /wp:paragraph --><!-- wp:paragraph --><p>{} publications ont été identifiées. {} publications ont été retenues pour un taux d\'inclusion de {:.1f}%. Le kappa entre les réviseurs était de {:.3f}.</p><!-- /wp:paragraph --><!-- wp:paragraph --><p>Les publications suivantes ont été retenues:</p><!-- /wp:paragraph --><!-- wp:list --><ul>{}</ul><!-- /wp:list --><!-- wp:spacer {{"height":40}} --><div style="height:40px" aria-hidden="true" class="wp-block-spacer"></div><!-- /wp:spacer --><p><a name="{}summary"></a></p><!-- wp:heading --><h2 id="{}summary">Summary</h2><!-- /wp:heading --><!-- wp:paragraph --><p>This Impact Briefing covers publications from {} to {}.</p><!-- /wp:paragraph --><!-- wp:paragraph --><p>{} publications were identified. {} publications were selected, for an inclusion rate of {:.1f}%. Inter-rater kappa was {:.3f}.</p><!-- /wp:paragraph --><!-- wp:paragraph --><p>The following publications were selected:</p><!-- /wp:paragraph --><!-- wp:list --><ul>{}</ul><!-- /wp:list --><!-- wp:spacer {{"height":40}} --><div style="height:40px" aria-hidden="true" class="wp-block-spacer"></div><!-- /wp:spacer --><!-- wp:heading --><h2>Publications</h2><!-- /wp:heading -->{}'

    briefing_update_pub_template = '<!-- wp:spacer {{"height":40}} --><div style="height:40px" aria-hidden="true" class="wp-block-spacer"></div><!-- /wp:spacer --><p><a name="{}"></a></p><!-- wp:heading {{"level":3}} --><h3 id="{}">{}</h3><!-- /wp:heading --><!-- wp:paragraph {{"fontSize":"small"}} --><p class="has-small-font-size">PMID: <a rel="noreferrer noopener" href="https://pubmed.ncbi.nlm.nih.gov/{}/" target="_blank">{}</a>{}<br><em>{}</em>, <em>{}</em></p><!-- /wp:paragraph -->{}<!-- wp:paragraph {{"fontSize":"small"}} --><p class="has-small-font-size">{}</p><!-- /wp:paragraph --><!-- wp:paragraph --><p><a href="#{}resume">Retour au résumé</a> - <a href="#{}summary">Return to summary</a></p><!-- /wp:paragraph -->'

    briefing_post_content = briefing_update_text_template.format(
        datetime.today().strftime('%Y%m%d'),
        datetime.today().strftime('%Y%m%d'),
        datetime.today().strftime('%Y%m%d'),
        start_date, 
        end_date, 
        selected_extraction_df.at[0, 'n_results'], 
        len(ds), 
        (len(ds) / selected_extraction_df.at[0, 'n_results']) *100, 
        cohen_kappa_score(current_extraction_df['rating1'].astype(int), current_extraction_df['rating2'].astype(int)), 
        ''.join(['<li><a href="#{}">{}</a> - {}</li>'.format(
            pmid, 
            data['title'],
            ' - '.join([tag for tag in data['machine_learning_tags_fr']]) 
            ) for pmid, data in ds.items()]),
        datetime.today().strftime('%Y%m%d'),
        datetime.today().strftime('%Y%m%d'),
        start_date, 
        end_date, 
        selected_extraction_df.at[0, 'n_results'], 
        len(ds), 
        (len(ds) / selected_extraction_df.at[0, 'n_results']) *100, 
        cohen_kappa_score(current_extraction_df['rating1'].astype(int), current_extraction_df['rating2'].astype(int)), 
        ''.join(['<li><a href="#{}">{}</a> - {}</li>'.format(
            pmid, 
            data['title'],
            ' - '.join([tag for tag in data['machine_learning_tags_eng']]) 
            ) for pmid, data in ds.items()]),
        ''.join([briefing_update_pub_template.format(
            pmid,
            pmid,
            data['title'],
            pmid,
            pmid,
            '     doi: <a rel="noreferrer noopener" href="https://dx.doi.org/{}" target="_blank">{}</a>'.format(data['doi'], data['doi']) if data['doi'] != '' else '',
            data['journal'],
            data['pubdate'],
            ' '.join([' '.join(['<!-- wp:paragraph --><p><strong>'+l+'</strong>',t+'</p><!-- /wp:paragraph -->']) for l,t in zip(data['labels'], data['texts']) if l != 'DISCLAIMER']),
            ', '.join(data['authors'])+'.',
            datetime.today().strftime('%Y%m%d'),
            datetime.today().strftime('%Y%m%d'),
            ) for pmid, data in ds.items()]),
        )
    briefing_post_title='Impact Briefing: {}'.format(datetime.today().strftime('%Y/%m/%d'))
    response = requests.post(post_url, headers=header, data={'title':briefing_post_title, 'content':briefing_post_content, 'categories':categories_briefing})
    print('Newsletter post response: {}'.format(response))

# MAIN

if __name__ == '__main__':

    with open(FILEPATH + '/credentials/spreadsheet_ids.json', mode='r') as file:
        spreadsheet_ids = json.load(file)

    if GOOGLE_SPREADSHEET_ID == 'REAL':
        google_spreadsheet_id = spreadsheet_ids['real_google_spreadsheet_id']
    else:
        google_spreadsheet_id = spreadsheet_ids['test_google_spreadsheet_id']

    ratings_sheet, predictions_sheet = get_google_sheets(google_spreadsheet_id, DATA_SHEET_NAME, PREDICTION_SHEET_NAME)

    with open(FILEPATH + '/credentials/pubmed_credentials.json', mode='r') as file:
        pubmed_credentials = json.load(file)
    
    with open(FILEPATH + '/credentials/wordpress_credentials.json', mode='r') as file:
        wordpress_token = json.loads((file.read()))

    site = 'impactpharmacie.net'
    post_url = 'https://public-api.wordpress.com/rest/v1.1/sites/{}/posts/new'.format(site)    
    header = {
        'Authorization': 'Bearer {}'.format(wordpress_token['access_token'])}

    extraction_log_df, selected_extraction_df, ratings_df, included_df, current_extraction_df, pmids = verify_and_validate_data(LOCAL_LOG_RELPATH, LOCAL_DATA_RELPATH, ratings_sheet, START_DATE, END_DATE)
    tokenizer = prepare_tokenizer()
    predictions = prepare_predictions_dict(included_df)
    tag_columns_to_use = []
    for model_string, t_dict in TAGS_TO_USE.items():
        model_version = t_dict['version']
        tag_columns_to_use.append(t_dict['column'])
        predictions = make_predictions(tokenizer, FILEPATH, model_string, model_version, included_df, predictions)
    prediction_df = convert_prediction_dict_to_df(predictions)
    prediction_tags_eng, prediction_tags_fr, prediction_tags_all = make_prediction_tags(prediction_df, tag_columns_to_use, TRANSLATION_DICT)
    ds = retrieve_pubmed_data(pmids, pubmed_credentials)
    verify_pubmed_retrieval(ds)
    ds = rebuild_dataset(ds, prediction_tags_fr, prediction_tags_eng, prediction_tags_all, pubmed_credentials)
    ds = verify_and_filter_dataset(ds)
    update_prediction_local_data(LOCAL_PREDICTIONS_RELPATH, prediction_df)
    update_prediction_google_sheet(predictions_sheet, prediction_df)
    update_ratings_local_data(LOCAL_DATA_RELPATH, ratings_sheet)
    publications_posts(ds, post_url, header)
    french_update_post(MONTH_NAMES_FR, START_DATE, END_DATE, selected_extraction_df, extraction_log_df, current_extraction_df, ratings_df, ds, post_url, header)
    english_update_post(MONTH_NAMES_ENG, START_DATE, END_DATE, selected_extraction_df, extraction_log_df, current_extraction_df, ratings_df, ds, post_url, header)
    make_newsletter_post(START_DATE, END_DATE, selected_extraction_df, extraction_log_df, current_extraction_df, ds, post_url, header)
    print('DONE !')





    