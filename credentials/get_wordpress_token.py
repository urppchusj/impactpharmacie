import json
import requests

from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import WebApplicationClient


scope=['posts']
redirect_uri = 'http://localhost/'
auth_url = 'https://public-api.wordpress.com/oauth2/authorize'
token_url = 'https://public-api.wordpress.com/oauth2/token'

with open('./credentials/wordpress_ids.json', mode='r') as file:
    wordpress_ids = json.load(file)

client_id = wordpress_ids['client_id']
client_secret = wordpress_ids['client_secret']

oauth = OAuth2Session(client_id, redirect_uri=redirect_uri, scope=scope)
authorization_url, state = oauth.authorization_url(auth_url)

print('Please go to {} and authorize access.'.format(authorization_url))
authorization_response = input('Enter the code: ')

client = WebApplicationClient(client_id)

token_params = {
    'client_id':client_id,
    'redirect_uri':redirect_uri,
    'client_secret':client_secret,
    'code':authorization_response,
    'grant_type':'authorization_code'
}

response = requests.post(token_url,data=token_params)
json_token = response.json()

with open('./credentials/wordpress_credentials.json', mode='w') as file:
    json.dump(json_token,file)
