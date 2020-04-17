# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 23:42:41 2020

@author: Evian Zhou
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import math
import os

import talib
from talib import MA_Type

import matplotlib.pyplot as plt

stock_universe_path= r'C:\Users\Evian Zhou\Documents\Python\TRIN\HSI constituents.csv'
stock_hist_data_path = r'C:\Users\Evian Zhou\Documents\Python\TRIN\Hist_data' + '\\'


start_date = '2015-01-01'
end_date = datetime.datetime.today().strftime("%Y-%m-%d")


def variance_calculator(series,series_average,win_len):
	sma = win_len
	temp = series.subtract(series_average)
	temp2 = temp.apply(lambda x: x**2)
	temp3 = temp2.rolling(sma).mean()
	sigma = temp3.apply(lambda x : math.sqrt(x))  
	return sigma


def Download_stock_price(stock_universe_path, stock_hist_data_path):
    stock_universe = pd.read_csv(stock_universe_path)
    stock_universe = stock_universe.values.tolist()
    
    print("Start to download historical data...")
    print("Start Date: " + start_date)
    print("End Date: " + end_date)

    for i in stock_universe:
        
        stock_OHLC = yf.download(i, start=start_date, end=end_date)
        stock_OHLC['Price_Chg'] = stock_OHLC['Adj Close'].diff()
        
        output_filename = stock_hist_data_path + i[0] + '_' + end_date + '.csv'
        #print(file_name)
        stock_OHLC.to_csv(output_filename)
        
    print("Download Data complete!")
    

def get_advancing_declining_stocks_count(target_date):
    
    advancing_declining_stocks_count = pd.DataFrame(columns=['Date', 'Advancing_Stocks_Count', 'Declining_stocks_count'])
    advancing_declining_stocks_count.set_index('Date')
    
    target_sources = os.listdir(stock_hist_data_path)
    target_sources = [s for s in target_sources if target_date in s]
    
    for file in target_sources:
        stock_OHLC = pd.read_csv(stock_hist_data_path + file)
        
        check= lambda 1 : stock_OHLC['Price_Chg'] > 0 else 0
        print(check)
    


    
    
def main():
    Download_stock_price(stock_universe_path, stock_hist_data_path)

    get_advancing_declining_stocks_count(end_date)
    
if __name__ == '__main__':

        main()