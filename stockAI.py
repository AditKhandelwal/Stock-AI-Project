import requests
import json

# find way to hide secret.txt
file = open("tokens.txt")
content = file.readlines()


# use your own account number, password and appid to authenticate (assigned from tokens.txt)
account = str(content[0]).strip()
password = content[1].strip()
appid = content[2].strip()

# Build auth url
authurl = "https://ftl.fasttrack.net/v1/auth/login?account=" + account + "&pass=" + password + "&appid=" + appid 

# make authentication request
authresponse = requests.request("GET", authurl)

# parse result and extract token
a = json.loads(authresponse.text)

token = a['token']


print(a)