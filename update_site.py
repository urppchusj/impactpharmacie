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
from urllib.parse import urlparse

# GLOBAL VARS
FILEPATH = '.'
GOOGLE_SPREADSHEET_ID = 'REAL' # 'REAL' OR 'TEST'
DATA_SHEET_NAME = 'data' # NAME OF DATA SHEET IN SPREADSHEET
LOG_SHEET_NAME = 'extraction_log' # NAME OF LOG SHEET IN SPREADSHEET
PREDICTION_SHEET_NAME = 'tag_predictions' # NAME OF PREDICTIONS SHEET IN SPREADSHEET
LOCAL_DATA_RELPATH = '/data/second_gen/ratings.csv'
LOCAL_LOG_RELPATH = '/data/second_gen/extraction_log.csv'
LOCAL_PREDICTIONS_RELPATH = '/data/second_gen/predictions.csv'
BASELINE_INCLUSION_MODEL_TO_USE = 'inclusion_biobert' # NAME (KEY) OF BASELINE INCLUSION MODEL IN INCLUSION_MODELS_TO_USE (SEE update_data.py FILE) ON WHICH INCLUSION SUGGESTIONS ARE BASED
ORIGINAL_START_DATE = '2021/11/07' # FORMAT 'YYYY/MM/DD'
START_DATE = '2025/07/06' # FORMAT 'YYYY/MM/DD'
END_DATE = '2025/07/19' # FORMAT 'YYYY/MM/DD'
SEARCH_QUERY = 'pharmacists[All Fields] OR pharmacist[All Fields] OR pharmacy[title]' # PUBMED QUERY STRING
MAX_PUBMED_TRIES = 10 # NUMBER OF MAXIMUM PUBMED QUERY TRIES BEFORE GIVING UP
ABSTRACT_SECTIONS_TO_EXCLUDE = ['DISCLAIMER', 'DISCLOSURE', 'DISCLOSURES'] # List of abstract labels that will be excluded from data 
WHERE_TO_PUBLICIZE = ['linkedin'] # List of services to publicize newsletter posts. So far: 'linkedin' // *** currently disabled due to bugs ***
TAGS_TO_USE = {'design':{'column':'design_pred', 'version':1}, 'field':{'column':'field_ground_truth', 'version':'0.1'}, 'setting':{'column':'setting_ground_truth','version':'0.1'}} # Dict with model strings as keys, values are dicts with column to use in dataframe as the first key and value and model version as second key and value
DESIGN_LABEL_TRANSLATIONS = {'Study':'Étude', 'Systematic review or meta-analysis':'Revue systématique ou méta-analyse'}
FIELDS_LABELS_TRANSLATIONS = {'Anticoagulation':'Anticoagulation', 'Cardiology':'Cardiologie', 'Critical care':'Soins critiques', 'Diabetes':'Diabète', 'Emergency medicine':'Urgence', 'Geriatric':'Gériatrie', 'Infectious diseases':'Infectiologie', 'Oncology':'Oncologie', 'Palliative care':'Soins palliatifs', 'Pneumology':'Pneumologie', 'Maternal / pediatric / neonatal':'Soins mère-enfant / pédiatrie / néonatologie', 'Psychiatric':'Psychiatrie', 'Solid organ transplantation':'Transplantation', 'Other':'Autre'}
SETTING_LABELS_TRANSLATIONS = {'Ambulatory':'Ambulatoire', 'Community':'Communautaire', 'Inpatient':'Établissement', 'Nursing home':'Soins de longue durée', 'Other':'Autre'}
POTENTIALLY_PREDATORY_ENG_LABEL = 'Potentially predatory journal or publisher'
OTHER_LABELS_TRANSLATIONS = {POTENTIALLY_PREDATORY_ENG_LABEL:'Journal ou éditeur potentiellement prédateur'}
POTENTIALLY_PREDATORY_TEMPLATE = '<!-- wp:paragraph {"fontSize":"small"} --><p class="has-small-font-size"><a href="https://impactpharmacie.net/journaux-et-editeurs-potentiellement-predateurs/" target="_blank" rel="noreferrer noopener">Article publié dans un journal ou par un éditeur potentiellement prédateur</a> - <a href="https://impactpharmacie.net/potentially-predatory-journals-and-publishers/" target="_blank" rel="noreferrer noopener">Paper published in a potentially predatory journal or by a potentially predatory publisher.</a></p><!-- /wp:paragraph -->'
TRANSLATION_DICT = {**DESIGN_LABEL_TRANSLATIONS, **FIELDS_LABELS_TRANSLATIONS, **SETTING_LABELS_TRANSLATIONS}
MONTH_NAMES_ENG = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
MONTH_NAMES_FR = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
PUBLISHER_LIST_URL = 'https://beallslist.net/'
JOURNALS_LIST_URL = 'https://beallslist.net/standalone-journals/'

# FUNCTIONS

def get_google_sheets(google_spreadsheet_id, data_sheet_name, prediction_sheet_name):
    credentials_filepath = FILEPATH + '/credentials/credentials.json'
    authorized_user_filepath = FILEPATH + '/credentials/authorized_user.json'
    try:
        gc = gspread.oauth(
            credentials_filename = credentials_filepath,
            authorized_user_filename = authorized_user_filepath
        )
        sht = gc.open_by_key(google_spreadsheet_id)
        ratings_sheet = sht.worksheet(data_sheet_name)
    except:
        if os.path.exists(authorized_user_filepath):
            os.remove(authorized_user_filepath)
        gc = gspread.oauth(
            credentials_filename = credentials_filepath,
            authorized_user_filename = authorized_user_filepath
        )
        sht = gc.open_by_key(google_spreadsheet_id)
        ratings_sheet = sht.worksheet(data_sheet_name)
    predictions_sheet = sht.worksheet(prediction_sheet_name)
    return ratings_sheet, predictions_sheet

def simplify_domain(domain):
    if len(urlparse(domain).hostname.split('.')) > 2:
        host = '.'.join(urlparse(domain).hostname.split('.')[1:])
    else:
        host = urlparse(domain).hostname
    return host

def take_and_validate_response():
    response = input('Accept ? (y/n)')
    if response not in ['y', 'n']:
        print('Please answer "y" or "n".')
        response = take_and_validate_response()
    return(response)

def validate_list_changes(old_list, new_list):

    validated_list = []

    print('Checking if new elements...')
    for el in new_list:
        if el in old_list:
            validated_list.append(el)
        else:
            print('Element {} was added to the list.'.format(el))
            response = take_and_validate_response()
            if response == 'y':
                validated_list.append(el)
            else:
                pass
    
    print('Checking if elements were removed...')
    for el in old_list:
        if el in new_list:
            pass
        else:
            print('Element {} was removed from the list.'.format(el))
            response = take_and_validate_response()
            if response == 'y':
                pass
            else:
                validated_list.append(el)

    return(validated_list)

def get_and_update_predatory_lists(publisher_list_url, journals_list_url):

    print('Updating potentially predatory journals and publishers...')
    
    def get_soup(list_url):
        page = requests.get(list_url)
        soup = BeautifulSoup(page.content, 'html.parser')
        return soup

    def get_domain_list(soup):
        the_list = []
        for li in soup.find(id='main').findAll('li'):
            link = li.find('a')
            try: 
                domain = link['href']
                the_list.append(domain)
            except:
                pass
        return the_list

    publisher_soup = get_soup(publisher_list_url)
    journals_soup = get_soup(journals_list_url)
    publisher_list = get_domain_list(publisher_soup)
    journals_list = get_domain_list(journals_soup)

    publisher_list_cleaned = [simplify_domain(href) for href in publisher_list]
    journals_list_cleaned = [simplify_domain(href) for href in journals_list]

    if os.path.exists(FILEPATH + '/data/second_gen/potentially_predatory_publishers.pkl'):
        with open(FILEPATH + '/data/second_gen/potentially_predatory_publishers.pkl', mode='rb') as file:
            old_publishers_list = pickle.load(file)
        print('Checking if predatory publishers list changed.')
        validated_publishers_list = validate_list_changes(old_publishers_list, publisher_list_cleaned)
    else:
        print('Previous predatory publishers list does not exist, no comparison done.')
        validated_publishers_list = publisher_list_cleaned
    with open(FILEPATH + '/data/second_gen/potentially_predatory_publishers.pkl', mode='wb') as file:
            pickle.dump(validated_publishers_list, file)

    if os.path.exists(FILEPATH + '/data/second_gen/potentially_predatory_journals.pkl'):
        with open(FILEPATH + '/data/second_gen/potentially_predatory_journals.pkl', mode='rb') as file:
            old_journals_list = pickle.load(file)
        print('Checking if predatory journals list changed.')
        validated_journals_list = validate_list_changes(old_journals_list, journals_list_cleaned)
    else:
        print('Previous predatory journals list does not exist, no comparison done.')
        validated_journals_list = journals_list_cleaned
    with open(FILEPATH + '/data/second_gen/potentially_predatory_journals.pkl', mode='wb') as file:
            pickle.dump(validated_journals_list, file)

    return(validated_publishers_list, validated_journals_list)

def verify_and_validate_data(local_log_relpath, local_data_relpath, ratings_sheet, start_date, end_date):

    extraction_log_df = pd.read_csv(FILEPATH + local_log_relpath, index_col=0).fillna('')
    extraction_log_df['pmids'] = extraction_log_df['pmids'].map(lambda x:x.split(', '))

    old_ratings_df = pd.read_csv(FILEPATH + local_data_relpath, index_col=0, dtype=str).fillna('')
    new_ratings_df = pd.DataFrame(ratings_sheet.get_all_records())
    ratings_df = pd.concat([old_ratings_df, new_ratings_df[1:]], ignore_index=True)[1:]
    ratings_df['rating_final'] = ratings_df.apply(lambda x: x['rating_consensus'] if x['rating_consensus'] != '' else x['rating1'] if ((x['rating1'] == x['rating2']) and (x['consensus_reason'] == '') and (x['rating1'] != '')) else 'error', axis=1)
    ratings_df['design_final'] = ratings_df['design_ground_truth']
    ratings_df['field_final'] = ratings_df.apply(lambda x: x['field_ground_truth_consensus'] if x['field_ground_truth_consensus'] != '' else x['field_ground_truth_1'] if ((x['field_ground_truth_1'] == x['field_ground_truth_2']) and (x['field_ground_truth_consensus_reason'] == '') and (x['field_ground_truth_1'] != '')) else '' if (x['rating_final'] == 0 or x['rating_final'] == '0') else 'error', axis=1)
    ratings_df['setting_final'] = ratings_df.apply(lambda x: x['setting_ground_truth_consensus'] if x['setting_ground_truth_consensus'] != '' else x['setting_ground_truth_1'] if ((x['setting_ground_truth_1'] == x['setting_ground_truth_2']) and (x['setting_ground_truth_consensus_reason'] == '') and (x['setting_ground_truth_1'] != '')) else '' if (x['rating_final'] == 0 or x['rating_final'] == '0') else 'error', axis=1)

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
    print('')
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
            print('ERROR, PubMed query error for pmid {} after {} tries, verify settings'.format(pmid, ntries))
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
        title, labels, texts, authors, journal, doi, pubdate, domain = process_single_pmid_data(pmid, data, pubmed_credentials)
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
        ds[pmid]['other_tags_fr'] = []
        ds[pmid]['other_tags_eng'] = []
        ds[pmid]['other_tags_all'] = []
        ds[pmid]['domain'] = domain

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
                try:
                    t = element_pmdata.find_all('BookTitle')
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
            try:
                journal = element_pmdata.find('MedlineTA').get_text()
            except:
                try:
                    journal = element_pmdata.find('PublisherName').get_text()
                except:
                    journal = ''
            doi_list = [e.get_text() for e in element_pmdata.find_all('ArticleId') if 'doi' in e.attrs.values()]
            if len(doi_list) > 0:
                doi = doi_list[0]
                try:
                    paper_page = requests.get('https://doi.org/{}'.format(doi))
                    host = simplify_domain(paper_page.url)
                    domain = host
                except: 
                    domain = 'unknown'
            else:
                doi = ''
                domain = 'unknown'
            pubdate = ' '.join([e.get_text() for e in element_pmdata.find('PubDate').children if e.name in ['Year', 'Month', 'Day']])
        except Exception as e:
            print('ERROR processing data for pmid {}, error: {} . Tried {} times, will retry for max {} times'.format(pmid, e, ntries, MAX_PUBMED_TRIES))
            if ntries <= MAX_PUBMED_TRIES:
                time.sleep(0.35)
                data['pmdata'] = get_pubmed_single_pmid(pmid, pubmed_credentials, ntries=ntries)
                title, labels, texts, authors, journal, doi, pubdate, domain = process_single_pmid_data(pmid, data, pubmed_credentials)
            else:
                print('ERROR processing data for pmid {} after {} tries, verify settings'.format(pmid, ntries))
                quit()
    return title, labels, texts, authors, journal, doi, pubdate, domain

def verify_and_filter_dataset(ds, potentially_predatory_publishers, potentially_predatory_journals):
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

    n_unknown_domains = 0
    for key,data in filtered_dataset.items():
        if data['domain'] == 'unkown':
            n_unknown_domains = n_unknown_domains + 1
    print('Number of papers with unkown domains: {}'.format(n_unknown_domains))

    def verify_if_predatory(dataset, comparison_list):
        for key, data in dataset.items():
            if data['domain'] in comparison_list:
                print('POTENTIALLY PREDATORY ELEMENT: PMID: {}  doi: {}   domain: {} . This paper will be tagged as potentially predatory.'.format(key, data['doi'], data['domain']))
                response = take_and_validate_response()
                if response == 'y':
                    if POTENTIALLY_PREDATORY_ENG_LABEL in ds[key]['other_tags_eng']:
                        pass
                    else:
                        ds[key]['other_tags_eng'].append(POTENTIALLY_PREDATORY_ENG_LABEL)
                        ds[key]['other_tags_fr'].append(OTHER_LABELS_TRANSLATIONS[POTENTIALLY_PREDATORY_ENG_LABEL])
                        ds[key]['other_tags_all'].append(POTENTIALLY_PREDATORY_ENG_LABEL)
                        ds[key]['other_tags_all'].append(OTHER_LABELS_TRANSLATIONS[POTENTIALLY_PREDATORY_ENG_LABEL])
            else: 
                pass
        return dataset

    filtered_dataset = verify_if_predatory(filtered_dataset,potentially_predatory_publishers)
    filtered_dataset = verify_if_predatory(filtered_dataset,potentially_predatory_journals)

    return (filtered_dataset)

def publications_posts(ds, post_url, header, abstract_sections_to_exclude, potentially_predatory_template, potentially_predatory_eng_label):
    categories = ['Publications']

    post_template = '<!-- wp:paragraph {{"fontSize":"small"}} --><p class="has-small-font-size">PMID: <a rel="noreferrer noopener" href="https://pubmed.ncbi.nlm.nih.gov/{}/" target="_blank">{}</a>{}<br><em>{}</em>, <em>{}</em></p><!-- /wp:paragraph -->{}{}<!-- wp:paragraph {{"fontSize":"small"}} --><p class="has-small-font-size">{}</p><!-- /wp:paragraph -->'

    for pmid, data in ds.items():
        post_content = post_template.format(
            pmid,
            pmid,
            '     doi: <a rel="noreferrer noopener" href="https://dx.doi.org/{}" target="_blank">{}</a>'.format(data['doi'], data['doi']) if data['doi'] != '' else '',
            data['journal'],
            data['pubdate'],
            '{}'.format(potentially_predatory_template if potentially_predatory_eng_label in data['other_tags_all'] else ''),
            ' '.join([' '.join(['<!-- wp:paragraph --><p><strong>'+l+'</strong>',t+'</p><!-- /wp:paragraph -->']) for l,t in zip(data['labels'], data['texts']) if l not in abstract_sections_to_exclude]),
            ', '.join(data['authors'])+'.'
            )
        title=data['title']
        post_tags_list = data['machine_learning_tags_all']
        post_data = {'slug': pmid, 'title':title, 'content':post_content, 'categories':categories, 'tags':','.join(post_tags_list), 'publicize':False}
        response = requests.post(post_url, headers=header, data=json.dumps(post_data))
        print('Publication update post response for PMID {}: {}'.format(pmid, response))

def french_update_post(month_names, start_date, end_date, selected_extraction_df, extraction_log_df, current_extraction_df, ratings_df, ds, post_url, header):
    categories_update = ['Mise à jour des données']
    update_post_template = '<!-- wp:paragraph --><p>Cette mise à jour couvre la période du {} au {}.</p><!-- /wp:paragraph --><!-- wp:paragraph --><p>{} publications ont été identifiées. {} ({:.1f}%) publications ont été filtrées par intelligence artificielle. {} ({:.1f}%) publications ont été révisées manuellement dont {} ({:.1f}%) ont été retenues. Le kappa entre les réviseurs était de {:.3f}.</p><!-- /wp:paragraph --><!-- wp:paragraph --><p>Les publications suivantes ont été retenues dans cette mise à jour:</p><!-- /wp:paragraph --><!-- wp:list --><ul>{}</ul><!-- /wp:list --><!-- wp:paragraph --><p>Depuis novembre 2021, {} publications ont été évaluées dont {} ({:.1f}%) ont été retenues. Le kappa entre les réviseurs pour toutes les publications évaluées est de {:.3f}.</p><!-- /wp:paragraph -->'

    kappa_df_current = current_extraction_df.loc[(current_extraction_df['rating1'] != '') & (current_extraction_df['rating2'] != '')]
    kappa_df_all = ratings_df.loc[(ratings_df['rating1'] != '') & (ratings_df['rating2'] != '')]

    update_post_content = update_post_template.format(
        start_date, 
        end_date, 
        '{:,}'.format(selected_extraction_df.at[0, 'n_results']).replace(',',' '), 
        len(current_extraction_df[current_extraction_df[BASELINE_INCLUSION_MODEL_TO_USE+'_suggestion'] == 'Exclude']),
        (len(current_extraction_df[current_extraction_df[BASELINE_INCLUSION_MODEL_TO_USE+'_suggestion'] == 'Exclude']) / selected_extraction_df.at[0, 'n_results']) *100,
        len(current_extraction_df[current_extraction_df[BASELINE_INCLUSION_MODEL_TO_USE+'_suggestion'] == 'Review']),
        (len(current_extraction_df[current_extraction_df[BASELINE_INCLUSION_MODEL_TO_USE+'_suggestion'] == 'Review']) / selected_extraction_df.at[0, 'n_results']) *100,
        len(ds), 
        (len(ds) / selected_extraction_df.at[0, 'n_results']) *100, 
        cohen_kappa_score(kappa_df_current['rating1'].astype(int), kappa_df_current['rating2'].astype(int)), 
        ''.join(['<li><a href="https://impactpharmacie.net/{}">{}</a> - {}</li>'.format(
            pmid, 
            data['title'],
            ' - '.join([tag for tag in data['machine_learning_tags_fr']]))
            for pmid, data in ds.items()]),
        len(ratings_df),
        ratings_df['rating_final'].astype(int).sum(), 
        (ratings_df['rating_final'].astype(int).sum() / len(ratings_df)) *100, 
        cohen_kappa_score(kappa_df_all['rating1'].astype(int), kappa_df_all['rating2'].astype(int))
        )
    update_post_title='Mise à jour du {} {} {}'.format(time.localtime()[2], month_names[time.localtime()[1]-1], time.localtime()[0])
    post_data = {'title':update_post_title, 'content':update_post_content, 'categories':categories_update, 'publicize':False}
    response = requests.post(post_url, headers=header, data=json.dumps(post_data))
    print('French update post response: {}'.format(response))

def english_update_post(month_names, start_date, end_date, selected_extraction_df, extraction_log_df, current_extraction_df, ratings_df, ds, post_url, header):
    categories_update = ['Data update']
    update_post_template = '<!-- wp:paragraph --><p>This update covers publications from {} to {}.</p><!-- /wp:paragraph --><!-- wp:paragraph --><p>{} publications were identified. {} ({:.1f}%) publications were filtered by machine learning. {} ({:.1f}%) publications were manually reviewed, of which {} ({:.1f}%) were selected. Inter-rater kappa was {:.3f}.</p><!-- /wp:paragraph --><!-- wp:paragraph --><p>The following publications were included in this update:</p><!-- /wp:paragraph --><!-- wp:list --><ul>{}</ul><!-- /wp:list --><!-- wp:paragraph --><p>Since November 2021, {} publications were evaluated. Among these, {} ({:.1f}%) were selected. Inter-rater kappa for all evaluated publications is {:.3f}.</p><!-- /wp:paragraph -->'

    kappa_df_current = current_extraction_df.loc[(current_extraction_df['rating1'] != '') & (current_extraction_df['rating2'] != '')]
    kappa_df_all = ratings_df.loc[(ratings_df['rating1'] != '') & (ratings_df['rating2'] != '')]

    update_post_content = update_post_template.format(
        start_date, 
        end_date, 
        '{:,}'.format(selected_extraction_df.at[0, 'n_results']).replace(',',' '),
        len(current_extraction_df[current_extraction_df[BASELINE_INCLUSION_MODEL_TO_USE+'_suggestion'] == 'Exclude']),
        (len(current_extraction_df[current_extraction_df[BASELINE_INCLUSION_MODEL_TO_USE+'_suggestion'] == 'Exclude']) / selected_extraction_df.at[0, 'n_results']) *100,
        len(current_extraction_df[current_extraction_df[BASELINE_INCLUSION_MODEL_TO_USE+'_suggestion'] == 'Review']),
        (len(current_extraction_df[current_extraction_df[BASELINE_INCLUSION_MODEL_TO_USE+'_suggestion'] == 'Review']) / selected_extraction_df.at[0, 'n_results']) *100,
        len(ds), 
        (len(ds) / selected_extraction_df.at[0, 'n_results']) *100, 
        cohen_kappa_score(kappa_df_current['rating1'].astype(int), kappa_df_current['rating2'].astype(int)),  
        ''.join(['<li><a href="https://impactpharmacie.net/{}">{}</a> - {}</li>'.format(
            pmid, 
            data['title'],
            ' - '.join([tag for tag in data['machine_learning_tags_eng']]) 
            ) for pmid, data in ds.items()]),
        len(ratings_df),
        ratings_df['rating_final'].astype(int).sum(), 
        (ratings_df['rating_final'].astype(int).sum() / len(ratings_df)) *100, 
        cohen_kappa_score(kappa_df_all['rating1'].astype(int), kappa_df_all['rating2'].astype(int))
        )
    update_post_title='Data update for {} {}, {}'.format(month_names[time.localtime()[1]-1], time.localtime()[2], time.localtime()[0])
    post_data = {'title':update_post_title, 'content':update_post_content, 'categories':categories_update, 'publicize':False}
    response = requests.post(post_url, headers=header, data=json.dumps(post_data))
    print('English update post response: {}'.format(response))

def make_newsletter_post(start_date, end_date, selected_extraction_df, extraction_log_df, current_extraction_df, ds, post_url, header, abstract_sections_to_exclude, where_to_publicize, potentially_predatory_template, potentially_predatory_eng_label):

    categories_briefing = ['Impact Briefing']

    briefing_update_text_template = '<!-- wp:paragraph --><p><a href="#{}summary">English</a></p><!-- /wp:paragraph --><!-- wp:spacer {{"height":40}} --><div style="height:40px" aria-hidden="true" class="wp-block-spacer"></div><!-- /wp:spacer --><p><a name="{}resume"></a></p><!-- wp:heading --><h2 id="{}resume">Résumé</h2><!-- /wp:heading --><!-- wp:paragraph --><p>Ce Impact Briefing couvre la période du {} au {}.</p><!-- /wp:paragraph --><!-- wp:paragraph --><p>{} publications ont été identifiées. {} ({:.1f}%) publications ont été filtrées par intelligence artificielle. {} ({:.1f}%) publications ont été révisées manuellement dont {} ({:.1f}%) ont été retenues. Le kappa entre les réviseurs était de {:.3f}.</p><!-- /wp:paragraph --><!-- wp:paragraph --><p>Les publications suivantes ont été retenues:</p><!-- /wp:paragraph --><!-- wp:list --><ul>{}</ul><!-- /wp:list --><!-- wp:spacer {{"height":40}} --><div style="height:40px" aria-hidden="true" class="wp-block-spacer"></div><!-- /wp:spacer --><p><a name="{}summary"></a></p><!-- wp:heading --><h2 id="{}summary">Summary</h2><!-- /wp:heading --><!-- wp:paragraph --><p>This Impact Briefing covers publications from {} to {}.</p><!-- /wp:paragraph --><!-- wp:paragraph --><p>{} publications were identified. {} ({:.1f}%) publications were filtered by machine learning. {} ({:.1f}%) publications were manually reviewed, of which {} ({:.1f}%) were selected. Inter-rater kappa was {:.3f}.</p><!-- /wp:paragraph --><!-- wp:paragraph --><p>The following publications were selected:</p><!-- /wp:paragraph --><!-- wp:list --><ul>{}</ul><!-- /wp:list --><!-- wp:spacer {{"height":40}} --><div style="height:40px" aria-hidden="true" class="wp-block-spacer"></div><!-- /wp:spacer --><!-- wp:heading --><h2>Publications</h2><!-- /wp:heading -->{}'

    briefing_update_pub_template = '<!-- wp:spacer {{"height":40}} --><div style="height:40px" aria-hidden="true" class="wp-block-spacer"></div><!-- /wp:spacer --><p><a name="{}"></a></p><!-- wp:heading {{"level":3}} --><h3 id="{}">{}</h3><!-- /wp:heading --><!-- wp:paragraph {{"fontSize":"small"}} --><p class="has-small-font-size">PMID: <a rel="noreferrer noopener" href="https://pubmed.ncbi.nlm.nih.gov/{}/" target="_blank">{}</a>{}<br><em>{}</em>, <em>{}</em></p><!-- /wp:paragraph -->{}{}<!-- wp:paragraph {{"fontSize":"small"}} --><p class="has-small-font-size">{}</p><!-- /wp:paragraph --><!-- wp:paragraph --><p><a href="#{}resume">Retour au résumé</a> - <a href="#{}summary">Return to summary</a></p><!-- /wp:paragraph -->'

    #brefing_update_publicize_template = 'Cette semaine, {} nouvelles publications ont été ajoutées à Impact Pharmacie. Abonnez-vous à notre liste de diffusion pour recevoir les résumés des publications sélectionnées à chaque semaine!'

    kappa_df_current = current_extraction_df.loc[(current_extraction_df['rating1'] != '') & (current_extraction_df['rating2'] != '')]

    briefing_post_content = briefing_update_text_template.format(
        datetime.today().strftime('%Y%m%d'),
        datetime.today().strftime('%Y%m%d'),
        datetime.today().strftime('%Y%m%d'),
        start_date, 
        end_date, 
        selected_extraction_df.at[0, 'n_results'], 
        len(current_extraction_df[current_extraction_df[BASELINE_INCLUSION_MODEL_TO_USE+'_suggestion'] == 'Exclude']),
        (len(current_extraction_df[current_extraction_df[BASELINE_INCLUSION_MODEL_TO_USE+'_suggestion'] == 'Exclude']) / selected_extraction_df.at[0, 'n_results']) *100,
        len(current_extraction_df[current_extraction_df[BASELINE_INCLUSION_MODEL_TO_USE+'_suggestion'] == 'Review']),
        (len(current_extraction_df[current_extraction_df[BASELINE_INCLUSION_MODEL_TO_USE+'_suggestion'] == 'Review']) / selected_extraction_df.at[0, 'n_results']) *100,
        len(ds), 
        (len(ds) / selected_extraction_df.at[0, 'n_results']) *100, 
        cohen_kappa_score(kappa_df_current['rating1'].astype(int), kappa_df_current['rating2'].astype(int)), 
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
        len(current_extraction_df[current_extraction_df[BASELINE_INCLUSION_MODEL_TO_USE+'_suggestion'] == 'Exclude']),
        (len(current_extraction_df[current_extraction_df[BASELINE_INCLUSION_MODEL_TO_USE+'_suggestion'] == 'Exclude']) / selected_extraction_df.at[0, 'n_results']) *100,
        len(current_extraction_df[current_extraction_df[BASELINE_INCLUSION_MODEL_TO_USE+'_suggestion'] == 'Review']),
        (len(current_extraction_df[current_extraction_df[BASELINE_INCLUSION_MODEL_TO_USE+'_suggestion'] == 'Review']) / selected_extraction_df.at[0, 'n_results']) *100,
        len(ds), 
        (len(ds) / selected_extraction_df.at[0, 'n_results']) *100, 
        cohen_kappa_score(kappa_df_current['rating1'].astype(int), kappa_df_current['rating2'].astype(int)), 
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
            '{}'.format(potentially_predatory_template if potentially_predatory_eng_label in data['other_tags_all'] else ''),
            ' '.join([' '.join(['<!-- wp:paragraph --><p><strong>'+l+'</strong>',t+'</p><!-- /wp:paragraph -->']) for l,t in zip(data['labels'], data['texts']) if l not in abstract_sections_to_exclude]),
            ', '.join(data['authors'])+'.',
            datetime.today().strftime('%Y%m%d'),
            datetime.today().strftime('%Y%m%d'),
            ) for pmid, data in ds.items()]),
        )

    #brefing_update_publicize_content = brefing_update_publicize_template.format(
    #    len(ds)
    #)

    briefing_post_title='Impact Briefing: {}'.format(datetime.today().strftime('%Y/%m/%d'))
    post_data = {'title':briefing_post_title, 'content':briefing_post_content, 'categories':categories_briefing, 'publicize':False}
    response = requests.post(post_url, headers=header, data=json.dumps(post_data))
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

    predatory_publishers_list, predatory_journals_list = get_and_update_predatory_lists(PUBLISHER_LIST_URL, JOURNALS_LIST_URL)
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
    ds = verify_and_filter_dataset(ds, predatory_publishers_list, predatory_journals_list)
    update_prediction_local_data(LOCAL_PREDICTIONS_RELPATH, prediction_df)
    update_prediction_google_sheet(predictions_sheet, prediction_df)
    update_ratings_local_data(LOCAL_DATA_RELPATH, ratings_sheet)
    publications_posts(ds, post_url, header, ABSTRACT_SECTIONS_TO_EXCLUDE, POTENTIALLY_PREDATORY_TEMPLATE, POTENTIALLY_PREDATORY_ENG_LABEL)
    french_update_post(MONTH_NAMES_FR, START_DATE, END_DATE, selected_extraction_df, extraction_log_df, current_extraction_df, ratings_df, ds, post_url, header)
    english_update_post(MONTH_NAMES_ENG, START_DATE, END_DATE, selected_extraction_df, extraction_log_df, current_extraction_df, ratings_df, ds, post_url, header)
    make_newsletter_post(START_DATE, END_DATE, selected_extraction_df, extraction_log_df, current_extraction_df, ds, post_url, header, ABSTRACT_SECTIONS_TO_EXCLUDE, WHERE_TO_PUBLICIZE, POTENTIALLY_PREDATORY_TEMPLATE, POTENTIALLY_PREDATORY_ENG_LABEL)
    print('DONE !')





    