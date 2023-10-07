import requests
import json
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import pandas as pd
import numpy as np 
import yfinance as yf
from finta import TA
import matplotlib.pyplot as plt

# Asks the user to input a stock ticker and validates the ticker to make sure that there are 
# no invalid tickers being inputted. Also forces ticker to uppercase.
def userInputTicker():
    TICKER = input("Enter a ticker: ")
    if (len(TICKER) > 5 or len(TICKER) == 0):
        print("Stock symbols don't typically have more than 5 letters and definitely don't have 0 letters. Enter a valid ticker: ")
        userInputTicker()
    else:
        TICKER = yf.Ticker(TICKER.upper())
        return TICKER


# Get date for 14 datalines
def getPastDate():
    weekdayCount = 0
    weekendCount = 0
    current_date = datetime.now()

    past_date = current_date - timedelta(days=weekendCount)


    while(weekdayCount != 14):
        if past_date.weekday() < 5:  # Saturday (5) or Sunday (6)
            weekdayCount += 1
        weekendCount += 1
        past_date = current_date - timedelta(days=weekendCount)
        
    return past_date

print(getPastDate())



# Download data for TICKER
#tickerData = yf.download(str(userInputTicker()), start="20" + pastYear + "-" + pastMonth +  "-" + pastDay, end="20" + currYear + "-" + currMonth +  "-" + currDay)

# Print the first 5 rows of the data
# print(len(testData))
# print(testData.head(4))

#print(tickerData.loc[1])
# print(tickerData.iloc[3, 4])