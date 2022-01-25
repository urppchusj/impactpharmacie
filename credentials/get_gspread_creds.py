import gspread

gc = gspread.oauth(
    credentials_filename='./credentials/credentials.json',
    authorized_user_filename='./credentials/authorized_user.json'
)
sh = gc.open("Example spreadsheet")
print(sh.sheet1.get('A1'))