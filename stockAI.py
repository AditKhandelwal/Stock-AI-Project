import requests
import json
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import pandas as pd
import numpy as np 
from finta import TA
import matplotlib.pyplot as plt
import yfinance as yf

# Asks the user to input a stock ticker and validates the ticker to make sure that there are 
# no invalid tickers being inputted. Also forces ticker to uppercase.


def checkValidTicker():
    valid = False
    TICKER = str(input("Enter a ticker: ")).upper()
    with open("nasdaq.txt", "r") as allowedTickersNASDAQ:
        for line in allowedTickersNASDAQ:
            firstItem = line.split("|")[0].strip()
            if TICKER == firstItem.upper():
                return TICKER

    if not valid:
        with open("nyse.txt", "r") as allowedTickersNYSE:
            for line in allowedTickersNYSE:
                firstItem = line.split(",")[0].strip()
                if TICKER == firstItem.upper():
                    return TICKER

    # Prompt user again if invalid Ticker
    if(valid == False):
        print("Invalid Ticker. Please enter a valid ticker")
        return checkValidTicker()
    
    allowedTickersNASDAQ.close()
    allowedTickersNYSE.close()
    
    return TICKER



#Smoothens out the data given from Yahoo Finance since having erratic data will cause an error in the Machine Learning model.
def _exponential_smooth(data, alpha):
    return data.ewm(alpha = alpha).mean()

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

# # create pandas object containing data for certain stock
def getTickerData():
    ticker = str(checkValidTicker()).upper()
    currentDate = datetime.now()
    currDate = currentDate - timedelta(days=1)
    pastDate = getPastDate()

    # Assign current/past date values
    currYear, currMonth, currDay = currDate.strftime("%y %m %d").split()
    pastYear, pastMonth, pastDay = pastDate.strftime("%y %m %d").split()
    tickerData = yf.download(ticker, start="20" + pastYear + "-" + pastMonth +  "-" + pastDay, end="20" + currYear + "-" + currMonth +  "-" + currDay)
    return tickerData


tickerData = getTickerData()
print(tickerData.head(3))

tickerDataSmooth = _exponential_smooth(tickerData, 0.65)
print(tickerDataSmooth.head(3))
# Get data by index location
# print(tickerData.iloc[3, 4])
