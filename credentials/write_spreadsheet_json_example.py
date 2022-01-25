import json

# TEST SPREADSHEET BELOW
test_google_spreadsheet_id = 'YOUR TEST GOOGLE SPREADSHEET ID HERE'
# REAL SPREADSHEET BELOW
real_google_spreadsheet_id = 'YOUR REAL GOOGLE SPREADSHEET ID HERE'

jsonObj = {'test_google_spreadsheet_id':test_google_spreadsheet_id, 'real_google_spreadsheet_id':real_google_spreadsheet_id}

with open('./credentials/spreadsheet_ids.json', mode='w') as file:
    json.dump(jsonObj, file)

