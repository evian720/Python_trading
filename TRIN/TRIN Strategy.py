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
stock_processed_data_path = r'C:\Users\Evian Zhou\Documents\Python\TRIN\Processed_data' + '\\'

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
    
    target_sources = os.listdir(stock_hist_data_path)
    target_sources = [s for s in target_sources if target_date in s]
    
    
    combined_stock_OHLC = pd.concat([pd.read_csv(stock_hist_data_path + csv) for csv in target_sources])
    combined_stock_OHLC['Value_Traded'] = combined_stock_OHLC['Adj Close'] * combined_stock_OHLC['Volume']
    
    
    combined_stock_OHLC = combined_stock_OHLC[['Date', 'Price_Chg', 'Value_Traded']]
    
    combined_stock_OHLC.loc[combined_stock_OHLC['Price_Chg'] > 0, 'Up_Down'] = 'Up'
    combined_stock_OHLC.loc[combined_stock_OHLC['Price_Chg'] < 0, 'Up_Down'] = 'Down'
    combined_stock_OHLC = combined_stock_OHLC.drop(columns='Price_Chg')
    
    # Store the Raw Up Down data
    #combined_stock_OHLC.to_csv(stock_processed_data_path + 'tmp.csv')
    
    # Count Up and Down stock
    Up_Down_count = combined_stock_OHLC.groupby('Date')['Up_Down'].value_counts().unstack()
    Up_Down_count.rename(columns={'Down': 'Down_Count', 'Up': 'Up_Count'}, inplace=True)
    
    # Sum Up and Down value traded
    Up_Down_Value_traded = combined_stock_OHLC.pivot_table(index='Date', columns='Up_Down', values='Value_Traded', aggfunc='sum')
    Up_Down_Value_traded.rename(columns={'Down':'Down_Value_Traded', 'Up': 'Up_Value_Traded'}, inplace=True)
  
    
    # Combine
    Up_Down_DF = pd.concat([Up_Down_count, Up_Down_Value_traded], axis=1)
    
    Up_Down_DF.fillna(method='ffill')
    
    print('Up/Down Stock Count and Value Traded Sum complete!')
    
    # Add Index close
    
    # 
    Up_Down_DF.to_csv(stock_processed_data_path + 'Up_Down_Count_' + end_date + '.csv')
    
    return Up_Down_DF


def calculate_TRIN(Up_Down_DF):
    
    # AD Ratio and AD Volumne ratio
    AD_Ratio = Up_Down_DF['Up_Count'].divide(Up_Down_DF['Down_Count'])
    AD_Value_Traded_Ratio = Up_Down_DF['Up_Value_Traded'].divide(Up_Down_DF['Down_Value_Traded'])
    
    Up_Down_DF['TRIN'] = AD_Ratio.divide(AD_Value_Traded_Ratio)
    
    Up_Down_DF['TRIN_Log'] = Up_Down_DF['TRIN'].apply(lambda x: math.log(x))
    
    
    print('TRIN calculate complete!')
    Up_Down_DF.to_csv(stock_processed_data_path + 'TRIN_' + end_date + '.csv')
    
    return Up_Down_DF

def TRIN_strategy(TRIN):
    # Define variables
    sma = 22
    k=1.5
    l=2
    
    pro=0
    
    flag = 1
    buy_flag = False
    sell_flag = False
    
    transaction_start_price = 0
    
    abs_sl = 25
    
    profit=list()
    buy_sell=list()
    stoploss=list()
    trade_cause=list()
    
    TRIN['mAvg'] = TRIN['TRIN_Log'].rolling(sma).mean()
    TRIN['Prev_TRIN_log'] = TRIN['TRIN_Log'].shift(1)
    
    # Calculate Sigma
    sigma = variance_calculator(TRIN['TRIN_Log'], TRIN['mAvg'], sma)
    k_sigma = k * sigma
    l_sigma = l * sigma
    
    # Calculate Bollinger Band of TRIN
    TRIN['UBB'] = TRIN['mAvg'].add(k_sigma)
    TRIN['LBB'] = TRIN['mAvg'].subtract(k_sigma)
    TRIN['USL'] = TRIN['UBB'].add(l_sigma)
    TRIN['LSL'] = TRIN['LBB'].subtract(l_sigma)
    
    TRIN['Orders'] = pd.Series()
    TRIN_size = TRIN['TRIN_Log'].size
    
    TRIN.to_csv(stock_processed_data_path + 'TRIN_Stregety_' + end_date + '.csv')
    
    # Generate buy sell signal
    for i in range(TRIN_size):
        pro = 0
        futures_cost = TRIN['Adj Close'][i]
        TRIN = TRIN['TRIN_Log'][i]
        #TRIN_Prev = TRIN['Prev_TRIN_log'][i]
        
        UBB = TRIN['UBB'][i]
        LBB = TRIN['LBB'][i]
        mAvg = TRIN['mAvg'][i]
        USL = TRIN['USL'][i]
        LSL = TRIN['LSL'][i]
        
        UBB_cross = (TRIN > UBB) and (TRIN_Prev < UBB)
        LBB_cross = (TRIN < LBB) and (TRIN_Prev > LBB)
        mAvg_cross_up = (TRIN > mAvg) and (TRIN_Prev < mAvg)
        mAvg_cross_down = (TRIN < mAvg) and (TRIN_Prev > mAvg)
        USL_cross = (TRIN > USL) and (TRIN_Prev < USL)
        LSL_cross = (TRIN < LSL) and (TRIN_prev > LSL)
        
        

def add_price_column_to_TRIN(TRIN, ticker):
        
    ticker_prices = yf.download(ticker, start=start_date, end=end_date)
   
    output_filename = stock_hist_data_path + ticker + '_' + end_date + '.csv'

    
    ticker_prices.to_csv(output_filename)
    ticker_prices = pd.read_csv(output_filename)
    
    ticker_prices.set_index('Date', inplace=True)
    
    print("Download " + ticker + " prices complete!")
    
    TRIN = pd.concat([TRIN, ticker_prices['Adj Close']], axis=1)
    
    return TRIN


def main():
    
    #Download_stock_price(stock_universe_path, stock_hist_data_path)

    Up_Down_DF = get_advancing_declining_stocks_count(end_date)
    
    TRIN = calculate_TRIN(Up_Down_DF)
    
    TRIN = add_price_column_to_TRIN(TRIN, '^HSI')
    TRIN_strategy(TRIN)
    
if __name__ == '__main__':

        main()