import requests
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from datetime import datetime, timedelta
import pandas as pd
import numpy as np 
from finta import TA
import matplotlib.pyplot as plt
import yfinance as yf

indicators = ['RSI', 'MACD', 'STOCH','ADL', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']

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


    while(weekdayCount != 5000):
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

# This method evaluates the stock against the indicators to get the values for the indicators.
def getIndicatorData(data):
    for indicator in indicators:
        indicatorData = eval('TA.' + indicator + '(data)')
        if not isinstance(indicatorData, pd.DataFrame):
            indicatorData = indicatorData.to_frame()
        data = data.merge(indicatorData, left_index = True, right_index = True)
    data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace = True)

    data["ema50"] = data["Close"] / data["Close"].ewm(50).mean()
    data["ema21"] = data["Close"] / data["Close"].ewm(21).mean()
    data["ema14"] = data["Close"] / data["Close"].ewm(14).mean()
    data["ema5"] = data["Close"] / data["Close"].ewm(5).mean()

    data["normalizedVolume"] = data["Volume"] / data["Volume"].ewm(5).mean()

    del (data["Open"])
    del (data["High"])
    del (data["Low"])
    del (data["Volume"])
    del (data["Adj Close"])

    return data

# This method adds a new column called "Prediction" with an integer binary value for a boolean (1 is true and 0 is false)
# and is used as the "correct answer" for the model when testing against the test cases.
def pricePredict(data, lookAheadDays):
    prediction = (data.shift(-lookAheadDays)["Close"] >= data["Close"])
    # prediction = prediction.iloc[:-lookAheadDays]
    data["Prediction"] = prediction.astype(int)
    return data

# Setting up and training a RandomForestClassifier model
def trainingRandomForestModel(X_train, y_train):
    rf = RandomForestClassifier()
    paramsRF = {'n_estimators': [110, 130, 140, 150, 160, 180, 200]}
    rfGridSearch = GridSearchCV(rf, paramsRF, cv=5)
    rfGridSearch.fit(X_train, y_train)
    rfBestModel = rfGridSearch.best_estimator_
    return rfBestModel

# Setting up and training a KNNeighbor model
def trainingKNN(X_train, y_train):
    knn = KNeighborsClassifier()
    paramKNN = {'n_neighbors': np.arange(1, 22)}
    knnGridSearch = GridSearchCV(knn, paramKNN, cv=5)
    knnGridSearch.fit(X_train, y_train)
    knnBestModel = knnGridSearch.best_estimator_
    return knnBestModel

# Setting up an Ensemble Model and training it
def ensembleModel(rf_model, knn_model, X_train, y_train):
    estimators = [('knn', knn_model), ('rf', rf_model)]
    ensemble = VotingClassifier(estimators, voting='hard')
    ensemble.fit(X_train, y_train)
    return ensemble


def crossValidation(data):
    numTrainPoints = 5000
    lengthOfTrainTestSet = 5000

    rfRESULTS = []
    knnRESULTS = []
    ensembleRESULTS = []

    i = 0
    while True:
        df = data.iloc[i * numTrainPoints : (i * numTrainPoints) + lengthOfTrainTestSet]
        i+=1

        if len(df) < 40:
            break

        y = df["Prediction"]
        features = [x for x in df.columns if x not in ["Prediction"]]
        X = df[features]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 7 * len(X) // 10, shuffle=False)

        rf_model = trainingRandomForestModel(X_train, y_train)
        knn_model = trainingKNN(X_train, y_train)
        ensemble_model = ensembleModel(rf_model, knn_model, X_train, y_train)

        rfPrediction = rf_model.predict(X_test)
        knnPrediction = knn_model.predict(X_test)
        ensemblePrediction = ensemble_model.predict(X_test)

        print("Random Forest Classifier prediction is: ", rfPrediction)
        print("KNNeighbor prediction is: ", knnPrediction)
        print("Ensemble prediction is: ", ensemblePrediction)
        print("Truth values are: ", y_test.values)

        rfAccuracy = accuracy_score(y_test.values, rfPrediction)
        knnAccuracy = accuracy_score(y_test.values, knnPrediction)
        ensembleAccuracy = accuracy_score(y_test.values, ensemblePrediction)

        print(rfPrediction, knnPrediction, ensemblePrediction)
        rfRESULTS.append(rfAccuracy)
        knnRESULTS.append(knnAccuracy)
        ensembleRESULTS.append(ensembleAccuracy)

        print("Random Forest Classifier accuracy = " + str(sum(rfRESULTS) / len(rfRESULTS)))
        print("KNNeighbor accuracy = " + str(sum(knnRESULTS) / len(knnRESULTS)))
        print("Ensemble accuracy = " + str(sum(ensembleRESULTS) / len(ensembleRESULTS)))





tickerData = getTickerData()
tickerDataSmooth = _exponential_smooth(tickerData, 0.65)
tickerIndicatorData = getIndicatorData(tickerDataSmooth)
tickerIndicatorData = tickerIndicatorData.dropna()
tickerINDICATORANDPREDICT = pricePredict(tickerIndicatorData, 1)
del (tickerINDICATORANDPREDICT["Close"])
tickerINDICATORANDPREDICT = tickerINDICATORANDPREDICT.dropna()
# print(tickerINDICATORANDPREDICT.tail(60))
crossValidation(tickerINDICATORANDPREDICT)

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error


# tickerIndicatorData['EMA'] = tickerIndicatorData['Close'].rolling(window=30).mean()
# tickerIndicatorData.dropna(subset=['EMA', 'Close'], inplace=True)


# print(tickerIndicatorData.columns)
# # Split data into training and testing sets
# X = tickerIndicatorData[['EMA']]
# # Assuming X is a DataFrame
# y = tickerIndicatorData['Close']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train a Linear Regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Calculate the root mean squared error (RMSE) as a metric
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# print(f'Root Mean Squared Error: {rmse}')

# # Visualize the predictions
# plt.figure(figsize=(12, 6))
# plt.plot(tickerIndicatorData.index, tickerIndicatorData['Close'], label='Actual')
# plt.plot(X_test.index, y_pred, label='Predicted', linestyle='--')
# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()