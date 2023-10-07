import requests
import json
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import pandas as pd
import numpy as np 
import yfinance as yf
from finta import TA
import matplotlib.pyplot as plt

indicators = ['RSI', 'MACD', 'STOCH', 'ADL', 'ATR', ]

# Asks the user to input a stock ticker and validates the ticker to make sure that there are 
# no invalid tickers being inputted. Also forces ticker to uppercase.
def userInputTicker():
    TICKER = str(input("Enter a ticker: ")).upper()
    if (len(TICKER) > 5 or len(TICKER) == 0):
        print("Stock symbols don't typically have more than 5 letters and definitely don't have 0 letters.")
        userInputTicker()
    else:
        try:
            stock_info = yf.Ticker(TICKER)
            if stock_info.info:
                print(stock_info.info)
                return TICKER
            else:
                print(f"{TICKER} is not a valid stock ticker symbol.")
                userInputTicker()
        except ValueError:
            print(f"{TICKER} is not a valid stock ticker symbol.")
            userInputTicker()

print(userInputTicker())
#Smoothens out the data given from Yahoo Finance since having erratic data will cause an error in the Machine Learning model.
def _exponential_smooth(data, alpha):
    return data.ewm(alpha = alpha).mean()

# Get date for 14 datalines
# def getPastDate():
#     weekdayCount = 0
#     weekendCount = 0
#     current_date = datetime.now()

#     past_date = current_date - timedelta(days=weekendCount)


#     while(weekdayCount != 14):
#         if past_date.weekday() < 5:  # Saturday (5) or Sunday (6)
#             weekdayCount += 1
#         weekendCount += 1
#         past_date = current_date - timedelta(days=weekendCount)

#     return past_date

# # create pandas object containing data for certain stock
# def getTickerData():
#     ticker = userInputTicker()
#     currentDate = datetime.now()
#     currDate = currentDate - timedelta(days=1)
#     pastDate = getPastDate()

#     # Assign current/past date values
#     currYear, currMonth, currDay = currDate.strftime("%y %m %d").split()
#     pastYear, pastMonth, pastDay = pastDate.strftime("%y %m %d").split()
#     tickerData = yf.download(ticker, start="20" + pastYear + "-" + pastMonth +  "-" + pastDay, end="20" + currYear + "-" + currMonth +  "-" + currDay)
#     return tickerData


# tickerData = getTickerData()
# print(tickerData.head(14))

# tickerDataSmooth = _exponential_smooth(tickerData, 0.65)



# Download data for TICKER
# tickerData = yf.download(str(userInputTicker()), start="20" + pastYear + "-" + pastMonth +  "-" + pastDay, end="20" + currYear + "-" + currMonth +  "-" + currDay)

# Print the first 5 rows of the data
# print(len(testData))
# print(testData.head(4))

#print(tickerData.loc[1])
# print(tickerData.iloc[3, 4])