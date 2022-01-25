import json

client_id = 'YOUR_CLIENT_ID_HERE'
client_secret = 'YOUR_CLIENT_SECRET_HERE'

jsonObj = {'client_id':client_id, 'client_secret':client_secret}

with open('./credentials/wordpress_ids.json', mode='w') as file:
    json.dump(jsonObj, file)

