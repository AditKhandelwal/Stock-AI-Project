import requests
import json
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import pandas as pd
import numpy as np 
import yfinance as yf



# find way to hide secret.txt
file = open("tokens.txt")
content = file.readlines()


# use your own account number, password and appid to authenticate (assigned from tokens.txt)
APIKEY = str(content[0]).strip()
TICKER = input("Enter a ticker: ")
url = "https://api.polygon.io/v1/open-close/AAPL/2023-09-17?adjusted=true&apiKey=" + APIKEY

# response = requests.get(url)

# json_data = response.json()

# print(json_data)

now = datetime.now().day

def dataDays():
    current_date = datetime.now()
    for i in range(14):
    # Calculate the date i days ago
        temp = current_date - timedelta(days=i)
        day = temp.strftime("%d")
        month = temp.strftime("%m")
        url = "https://api.polygon.io/v1/open-close/" + TICKER + "/2023-" + str(day) + "-" + str(month) + "?adjusted=true&apiKey=" + APIKEY
        
        #Print out JSON
        response = requests.get(url)
        json_data = response.json()
        # if (json_data["status"] == "OK"):
            # print(json_data)
            # print("Date: " + str(json_data["from"]) + " | Open Price: " + str(json_data["open"]) + " | Close Price: " + str(json_data["close"]))

dataDays()

# url = "https://api.polygon.io/v1/open-close/" + TICKER + "/2023-0" + str(DATE) + "-17?adjusted=true&apiKey=" + APIKEY

# response = requests.get(url)

# json_data = response.json()

# print(json_data)

# print("Date: " + str(json_data["from"]) + " | Open Price: " + str(json_data["open"]) + " | Close Price: " + str(json_data["close"]))


    

