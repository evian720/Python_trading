# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:52:06 2020

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


"""
Definitions
"""
stock_universe_path= r'C:\Users\Evian Zhou\Documents\Python\Screnner\stock universe.csv'
index_universe_path = r'C:\Users\Evian Zhou\Documents\Python\Screnner\index universe.csv'

stock_hist_data_path = r'C:\Users\Evian Zhou\Documents\Python\Screnner\Hist_Data' + '\\'
stock_processed_data_path = r'C:\Users\Evian Zhou\Documents\Python\Screnner\Processed_Data' + '\\'


# Technical Analysis Data Output
stochastic_oscillator_path = r'C:\Users\Evian Zhou\Documents\Python\Screnner\Stochastic_Oscillator' + '\\'
bollinger_bands_path = r'C:\Users\Evian Zhou\Documents\Python\Screnner\Bollinger_Bands' + '\\'
MACD_path = r'C:\Users\Evian Zhou\Documents\Python\Screnner\MACD' + '\\'



start_date = '2015-01-01'
end_date = datetime.datetime.today().strftime("%Y-%m-%d")

"""
========================================================================
========================================================================
"""


def Download_stock_price(stock_universe_path, stock_hist_data_path):
    stock_universe = pd.read_csv(stock_universe_path)
    stock_universe = stock_universe.values.tolist()
    
    print("Start to download historical data...")
    print("Start Date: " + start_date)
    print("End Date: " + end_date)

    for i in stock_universe:
        
        stock_OHLC = yf.download(i, start=start_date, end=end_date)
        
        output_filename = stock_hist_data_path + i[0] + '_' + end_date + '.csv'
        #print(file_name)
        stock_OHLC.to_csv(output_filename)
        
    print("Download Data complete!")

"""
========================================================================
"""

def Add_indicator(stock_OHLC_Source_path, filename, stock_OHLC_Output_path):
    

    
    stock_OHLC = pd.read_csv(stock_OHLC_Source_path + filename)
    
    # SMA
    sma_period = 20
    lma_period = 50
    
    stock_OHLC['SMA'] = talib.SMA(stock_OHLC['Adj Close'], timeperiod = sma_period)
    stock_OHLC['LMA'] = talib.SMA(stock_OHLC['Adj Close'], timeperiod = lma_period)
    
    # RSI
    stock_OHLC['RSI14'] = talib.RSI(stock_OHLC['Adj Close'], timeperiod = 14)
    stock_OHLC['RSI28'] = talib.RSI(stock_OHLC['Adj Close'], timeperiod = 28)
    
    # Bollinger Bands
    stock_OHLC['Upper_band'], stock_OHLC['Middle_band'], stock_OHLC['Lower_band'] = talib.BBANDS(stock_OHLC['Close'], matype=MA_Type.T3)
    
    # Stochastic Oscillator
    stock_OHLC['SAR'] = talib.SAR(stock_OHLC['High'].values, stock_OHLC['Low'].values, acceleration=0.02, maximum=0.2)
    
    stock_OHLC['slowk'], stock_OHLC['slowd'] = talib.STOCH((stock_OHLC['High'].values), (stock_OHLC['Low'].values), (stock_OHLC['Close'].values)
                                    ,fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)


    stock_OHLC['fastk'], stock_OHLC['fastd'] = talib.STOCHF((stock_OHLC['High'].values), (stock_OHLC['Low'].values), (stock_OHLC['Close'].values)
                                     , fastk_period=5, fastd_period=3, fastd_matype=0)
    
    # AR BR
    arbr_timeperiod = 26
    
    stock_OHLC['HO'] = stock_OHLC['High'] - stock_OHLC['Open']
    stock_OHLC['OL'] = stock_OHLC['Open'] - stock_OHLC['Low']
    stock_OHLC['HCY'] = stock_OHLC['High'] - stock_OHLC['Close'].shift(1)
    stock_OHLC['CYL'] = stock_OHLC['Close'].shift(1) - stock_OHLC['Low']
    
    stock_OHLC['AR'] = talib.SUM(stock_OHLC['HO'], timeperiod=arbr_timeperiod ) / talib.SUM(stock_OHLC['OL'], timeperiod=arbr_timeperiod) * 100
    stock_OHLC['BR'] = talib.SUM(stock_OHLC['HCY'], timeperiod=arbr_timeperiod) / talib.SUM(stock_OHLC['CYL'], timeperiod=arbr_timeperiod) * 100
    
    
    # MACD
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    stock_OHLC['MACD'], stock_OHLC['MACD_Signal'], stock_OHLC['MACD_Hist'] = talib.MACD(stock_OHLC['Close'], fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL)
    stock_OHLC['MACD_Test'] = np.where(stock_OHLC['MACD'] > stock_OHLC['MACD_Signal'],1, 0)
    
    stock_OHLC.to_csv(stock_OHLC_Output_path + filename)


def Add_indicator_all(target_date):
    target_sources = os.listdir(stock_hist_data_path)
    target_sources = [s for s in target_sources if target_date in s]
    
    for filename in target_sources: 
        filepath = stock_hist_data_path + filename
        Add_indicator(stock_hist_data_path, filename, stock_processed_data_path)
        
        print("Add indicator for " + filename + " complete!")

"""
========================================================================
"""

def traing_signal_stochastic_oscillator(stock_OHLC_Source_path, filename):
    stock_OHLC = pd.read_csv(stock_OHLC_Source_path)
    
    stock_OHLC.index = stock_OHLC.Date
    stock_OHLC = stock_OHLC[['Date', 'Close', 'SAR', 'slowk', 'slowd', 'fastk','fastd']]
    
    stock_OHLC['Position'] = np.nan
    
    stock_OHLC.loc[ (stock_OHLC['SAR'] < stock_OHLC['Close']) & (stock_OHLC['fastd'] > stock_OHLC['slowd']) & (stock_OHLC['fastk'] > stock_OHLC['slowk']) ,'Position'] = 1
    stock_OHLC.loc[ (stock_OHLC['SAR'] > stock_OHLC['Close']) & (stock_OHLC['fastd'] < stock_OHLC['slowd']) & (stock_OHLC['fastk'] < stock_OHLC['slowk']) , 'Position'] = -1
    
    
    stock_OHLC['Position'] = stock_OHLC['Position'].fillna(method='ffill')
    stock_OHLC['Position'] = stock_OHLC['Position'].fillna(0)
        
    stock_OHLC.loc[ stock_OHLC['Position'].diff() > 0  ,'Stochastic_Oscillator_Action'] = 'Buy'
    stock_OHLC.loc[ stock_OHLC['Position'].diff() < 0  ,'Stochastic_Oscillator_Action'] = 'Sell'
    
    
    
    plot_x = stock_OHLC.Date
    plot_y_close = stock_OHLC.Close
    plot_y_SAR = stock_OHLC.SAR
    plot_y_action = stock_OHLC.Stochastic_Oscillator_Action
    
    plt.title('Stochasic Oscillator Actions')
    plt.xlabel('Close')
    plt.ylabel('Time')
    
    
    plt.plot(plot_x, plot_y_close, label='Close Price')
    plt.plot(plot_x, plot_y_SAR, label='SAR')
    
    Buy_actions_dates = stock_OHLC.loc[ ( stock_OHLC.Stochastic_Oscillator_Action == 'Buy'), 'Date']
    Sell_actions_dates = stock_OHLC.loc[ ( stock_OHLC.Stochastic_Oscillator_Action == 'Sell'), 'Date']
    for d in Buy_actions_dates:
        plt.axvline(d,color='red')
        
    for d in Sell_actions_dates:
        plt.axvline(d,color='green')
    
    stock_OHLC.to_csv(stochastic_oscillator_path + filename)
    
    return stock_OHLC


def traing_signal_stochastic_oscillator_all(target_date):
    target_sources = os.listdir(stock_processed_data_path)
    target_sources = [s for s in target_sources if target_date in s]
    
    #print(target_sources)
    
    for filename in target_sources: 
        filepath = stock_processed_data_path + filename
        traing_signal_stochastic_oscillator( filepath, filename)
        
        print("Add Stochastic Oscillator for " + filename + " complete!")


"""
========================================================================
========================================================================
"""

def trading_signal_bollinger_bands(stock_OHLC_Source_path, filename):
    
    # Value n for standard deviation
    n=60
    
    stock_OHLC = pd.read_csv(stock_OHLC_Source_path)
    stock_OHLC = stock_OHLC[['Date', 'Close', 'Volume', 'Upper_band', 'Middle_band', 'Lower_band']]
    
    stock_OHLC['Return'] = stock_OHLC['Close'].pct_change()
    stock_OHLC['Return_Std'] = stock_OHLC['Return'].rolling(window=n).std()
    
    # Add signal
    stock_OHLC['Position'] = 0
    
    stock_OHLC.loc[ (stock_OHLC['Upper_band'].shift(2)>stock_OHLC['Close'].shift(2)) & (stock_OHLC['Upper_band'].shift(1)<stock_OHLC['Close'].shift(1)) \
                   & (stock_OHLC['Return_Std'].shift(1)>stock_OHLC['Return'].shift(1)), 'Position']=1
        
    stock_OHLC.loc[ (stock_OHLC['Lower_band'].shift(2)<stock_OHLC['Close'].shift(2)) & (stock_OHLC['Lower_band'].shift(1)>stock_OHLC['Close'].shift(1)) \
                   & (-stock_OHLC['Return_Std'].shift(1)<stock_OHLC['Return'].shift(1)), 'Position']=-1
    
    print(stock_OHLC)

    
    
    """ 
    plot_x = stock_OHLC.Date
    plot_y_close = stock_OHLC.Close
    
    plt.title('Bollinger Bands Actions')
    plt.xlabel('Close')
    plt.ylabel('Time')
    
    
    plt.plot(plot_x, plot_y_close, label='Close Price')


    Buy_actions_dates = stock_OHLC.loc[ ( stock_OHLC.Position == 1), 'Date']
    Sell_actions_dates = stock_OHLC.loc[ ( stock_OHLC.Position == -1), 'Date']
    
    for d in Buy_actions_dates:
        plt.axvline(d,color='red')
        
    for d in Sell_actions_dates:
        plt.axvline(d,color='green')
    """
    
    # Return
    stock_OHLC['Strategy_return']=stock_OHLC['Position']*stock_OHLC['Return']
    trades = np.count_nonzero(stock_OHLC['Position'])
    
    
    # TEMPPPP
    sum_stragety_return = stock_OHLC['Strategy_return'].sum()
    print(sum_stragety_return)
    
    
    
    plt.plot((stock_OHLC['Strategy_return']+1).cumprod())
    plt.figtext(0.14,0.9,s='\n\nTrades:%i'%trades)
   
    plt.title(filename.replace('.csv', '.png'))
    
    plt.savefig(bollinger_bands_path + filename.replace('.csv', '.png'))
    plt.show()
    
    
    
    stock_OHLC.to_csv(bollinger_bands_path + filename)
    

def trading_signal_bollinger_bands_all(target_date):
    target_sources = os.listdir(stock_processed_data_path)
    target_sources = [s for s in target_sources if target_date in s]
    
    #print(target_sources)
    
    for filename in target_sources: 
        filepath = stock_processed_data_path + filename
        trading_signal_bollinger_bands( filepath, filename)
        
        print("Add Bollinger Bands for " + filename + " complete!")
        

"""
========================================================================
"""


def trading_signal_arbr(stock_OHLC_Source_path, filename):
    stock_OHLC = pd.read_csv(stock_OHLC_Source_path)
    stock_OHLC = stock_OHLC[['Date', 'Close', 'Volume', 'HO', 'OL', 'HCY', 'CYL', 'AR', 'BR']]
    
    
    stock_OHLC['Close'].plot(color='r',figsize=(14,5))
    stock_OHLC[['AR','BR']].plot(figsize=(14,5))
    
    plt.show()
    
    print(stock_OHLC)

"""
========================================================================
"""

def trading_signal_MACD(stock_OHLC_Source_path, filename):
    stock_OHLC = pd.read_csv(stock_OHLC_Source_path)
    stock_OHLC = stock_OHLC[['Date', 'Close', 'Volume', 'MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_Test']]
    print(stock_OHLC)

    fig1, ax = plt.subplots(2, sharex=True, figsize=(16, 8))
    ax[0].plot(stock_OHLC['Close'], label='Close')
    ax[0].legend(loc='upper left')
    ax[0].grid()
    

    ax[1].bar(stock_OHLC.index, stock_OHLC['MACD_Hist'] , width=1, label='Hist')
    ax[1].plot(stock_OHLC['MACD'], label='MACD')
    ax[1].plot(stock_OHLC['MACD_Signal'], label='MACD_Signal')
    plt.axhline(0, color='gray', linewidth=3, linestyle='-.' )
    plt.legend(loc='upper left')
    plt.grid()
    
    plt.suptitle(filename + ' Close and MACD(12,26,9)')
    
    plt.savefig(MACD_path + filename.replace('.csv', '.png'))
    
    plt.show()

def trading_signal_MACD_all(target_date):
    target_sources = os.listdir(stock_processed_data_path)
    target_sources = [s for s in target_sources if target_date in s]
    
    #print(target_sources)
    
    for filename in target_sources: 
        filepath = stock_processed_data_path + filename
        trading_signal_MACD( filepath, filename)
        
        print("Add MACD for " + filename + " complete!")


"""
========================================================================
========================================================================
"""


def main():
    Download_stock_price(stock_universe_path, stock_hist_data_path)
    Download_stock_price(index_universe_path, stock_hist_data_path)

    Add_indicator_all(end_date)
    #Add_indicator(stock_hist_data_path, '^HSI_2020-04-13.csv', stock_processed_data_path)
    
    traing_signal_stochastic_oscillator_all(end_date)
    
    
    #trading_signal_bollinger_bands(stock_processed_data_path + '0027.HK_2020-04-12.csv', '0027.HK_2020-04-12.csv')
    trading_signal_bollinger_bands_all(end_date)
    
    #trading_signal_arbr(stock_processed_data_path + '^HSI_2020-04-13.csv', '^HSI_2020-04-13.csv')
    #trading_signal_MACD(stock_processed_data_path + '0700.HK_2020-04-13.csv', '0700.HK_2020-04-13.csv')
    
    trading_signal_MACD_all(end_date)
    
if __name__ == '__main__':

        main()