import json

# TOOL NAME FOR PUBMED BELOW
pubmed_tool_name = 'YOUR TOOL NAME HERE'
# CONTACT EMAIL FOR PUBMED BELOW
pubmed_tool_email = 'YOUR CONTACT EMAIL HERE'

jsonObj = {'pubmed_tool_name':pubmed_tool_name, 'pubmed_tool_email':pubmed_tool_email}

with open('./credentials/pubmed_credentials.json', mode='w') as file:
    json.dump(jsonObj, file)

