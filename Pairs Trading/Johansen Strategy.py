# -*- coding: utf-8 -*-
"""
Created on Sat May  2 20:02:38 2020

@author: Evian Zhou
"""


# For data manupulation
import numpy as np
import pandas as pd

# For plotting
import matplotlib.pyplot as plt  


# For checking cointegration
from johansen import coint_johansen

# Yahoo Finance for download the stock OHLC prices
import yfinance as yf

# Basic system function
import datetime
import time
import os

# Define the dates and file location
start_date = '2015-01-01'
end_date = datetime.datetime.today().strftime("%Y-%m-%d")

stock_universe_path = r'C:\Users\Evian Zhou\Documents\Python\Pairs Trading\Pairs.csv'
stock_hist_data_path = r'C:\Users\Evian Zhou\Documents\Python Trading Output\Johansen Strategy\Hist_data' + '\\' + end_date + '\\'
stock_output_path = r'C:\Users\Evian Zhou\Documents\Python Trading Output\Johansen Strategy\Out_put' + '\\' + end_date + '\\'



# Define Johansen test period
Johansen_Period = 210
Lookback_Period = 90



# Create folder for Source Data
try:
    os.mkdir(stock_hist_data_path)
except OSError:
    print("Creation of the directory %s failed" % stock_hist_data_path)
else:
    print("Successfully created the directory %s " % stock_hist_data_path)

# Create folder for output
try:
    os.mkdir(stock_output_path)
except OSError:
    print("Creation of the directory %s failed" % stock_output_path)
else:
    print("Successfully created the directory %s " % stock_output_path)



def listToString(s):      
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 = str1 + '' +  ele   

    # return string   
    return str1  

def Download_stock_price(stock_universe_path, stock_hist_data_path):
    
    stock_universe = pd.read_csv(stock_universe_path)
    stock_universe = stock_universe.values.tolist()
    
    print("Start to download historical data...")
    print("Start Date: " + start_date)
    print("End Date: " + end_date)
    
    
    # Download prices for each stocks
    
    for i in stock_universe:
        stock_list = i[0].split(',')
        for single_stock in stock_list:
            stock_OHLC = yf.download(single_stock, start=start_date, end=end_date)
            output_filename = stock_hist_data_path + single_stock + '_' + end_date + '.csv'
            #print(file_name)
            stock_OHLC.to_csv(output_filename)
        
    print("Download Data complete!")
    
    

def Johansen_Strategy(stock_list):
    stock_list = stock_list.split(',')
    Stock_List_DF = pd.DataFrame()

    # Loop thru the list and create pairs dataframe
    for stock in stock_list:
        print('*** ' + stock)
        
        stock_source_path = stock_hist_data_path + stock + '_' + end_date + '.csv'
        
        stock_df = pd.read_csv(stock_source_path)
        stock_df.set_index(['Date'], inplace=True)
        stock_df = stock_df['Adj Close']
        stock_df.name = stock
        
        Stock_List_DF = pd.concat([Stock_List_DF,stock_df], axis=1)
        
        Stock_List_DF.dropna(inplace=True)
        
    
    print(Stock_List_DF.head())
    
    # Running Johansen test only on first 90 days of data
    # Storing the eigenvectors
    result = coint_johansen(Stock_List_DF[:Johansen_Period] ,0 ,1)
    d = result.evec
    ev= d[0]
    
    # Normalizing the eigenvectors
    ev = ev/ev[0]
    ev = ev.round(decimals=2)
    

    # Printing the mean reverting spread
    pairs_name = listToString(stock_list) 
    print('The Strategy for ' + listToString(stock_list) + ' :')
    
    print('---------------')
    for i in ev:
        print(i)
    print('---------------')
    
    
    # Calculate Spread
    Stock_List_DF['Spread'] = 0
    ev_counter = 0
    
    for column in Stock_List_DF:
        if column == 'Spread':
            break
        
        Stock_List_DF['Spread'] = Stock_List_DF['Spread'] + Stock_List_DF[column] * ev[ev_counter]
        ev_counter += 1
        
        
        
    # Calculate the upper and lower band
    # Moving Average and Moving Standard Deviation
    Stock_List_DF['Moving_Avg'] = Stock_List_DF.Spread.rolling(Lookback_Period).mean()
    Stock_List_DF['Moving_Std'] = Stock_List_DF.Spread.rolling(Lookback_Period).std()

    # Upper band and lower band
    Stock_List_DF['Upper_Band'] = Stock_List_DF.Moving_Avg + 0.5 * Stock_List_DF.Moving_Std
    Stock_List_DF['Lower_Band'] = Stock_List_DF.Moving_Avg - 0.5 * Stock_List_DF.Moving_Std    
        
    
    
    # Long entry and long exit
    # Long_entry is set to True value whenever the price falls below the lower band and False otherwise.
    # Long_exit is set to True value whenever the prices mean-reverts to the current moving average (prices >= moving average) and False otherwise.
    Stock_List_DF['Long_Entry'] = Stock_List_DF.Spread < Stock_List_DF.Lower_Band   
    Stock_List_DF['Long_Exit'] = Stock_List_DF.Spread >= Stock_List_DF.Moving_Avg
    
    # Short entry and short exit
    # Short entry is set to True value whenever the price rises above the upper band and False otherwise.
    # Short exit is set to True value whenever the prices mean-reverts to the current moving average (prices <= moving average) and False otherwise.
    Stock_List_DF['Short_Entry'] = Stock_List_DF.Spread > Stock_List_DF.Upper_Band   
    Stock_List_DF['Short_Exit'] = Stock_List_DF.Spread <= Stock_List_DF.Moving_Avg
    
    
    # Long positions and short positions
    # positions_long and positions_short columns is initialized with NaN values using np.nan.
    # 1 is assigned to positions_long when long_entry is True and 0 when long_exit is True.
    # Similarly, -1 and 0 are assigned to positions_short.
    
    Stock_List_DF['Positions_Long'] = np.nan  
    Stock_List_DF.loc[Stock_List_DF.Long_Entry,'Positions_Long']= 1  
    Stock_List_DF.loc[Stock_List_DF.Short_Exit,'Positions_Long']= 0  
  
    Stock_List_DF['Positions_Short'] = np.nan  
    Stock_List_DF.loc[Stock_List_DF.Short_Entry,'Positions_Short']= -1  
    Stock_List_DF.loc[Stock_List_DF.Short_Exit,'Positions_Short']= 0  

    # Fill NaN values
    Stock_List_DF = Stock_List_DF.fillna(method='ffill')  

    # Consolidate the positions
    Stock_List_DF['Positions'] = Stock_List_DF.Positions_Long + Stock_List_DF.Positions_Short
    
    Stock_List_DF['Strategy_Return'] = (Stock_List_DF.Spread - Stock_List_DF.Spread.shift(1))* Stock_List_DF.Positions.shift(1)
    Stock_List_DF['Accumulated_Strategy_Return'] = Stock_List_DF.Strategy_Return.cumsum()
    
    
    # Store the Dataframe
    Stock_List_DF.to_csv(stock_output_path + pairs_name + '_' + end_date + '.csv')
    
    
    # Plotting the spread
    fig, axs = plt.subplots(2, 1, constrained_layout=True,figsize=(16, 12))
    Stock_List_DF['Spread'].plot(ax=axs[0], lw=2, label='Spread')
    Stock_List_DF['Moving_Avg'].plot(ax=axs[0], lw=1,style='--', label='Spred Moving Avg')
    Stock_List_DF['Upper_Band'].plot(ax=axs[0], lw= 1,style='--', label='Upper Band')
    Stock_List_DF['Lower_Band'].plot(ax=axs[0], lw= 1,style='--', label='Lower Band')
    axs[0].set_xlabel('Date')
    axs[0].set_title('Spread')
    axs[0].set_ylabel('Spread')
    axs[0].grid()
    axs[0].legend(loc='upper left')

    
    Stock_List_DF['Accumulated_Strategy_Return'].plot(ax=axs[1], lw=1, label='Accumulated_Strategy_Return')
    axs[1].set_xlabel('Date')
    axs[1].set_title('Accumulated Strategy Return')
    axs[1].set_ylabel('Return')
    axs[1].grid()
    axs[1].legend(loc='upper left')
    
    fig.suptitle(pairs_name + ' ' + end_date, fontsize=16)
    fig.savefig(stock_output_path + pairs_name + '_' + end_date + '.png')

def main():
    #Download_stock_price(stock_universe_path, stock_hist_data_path)
    
    stock_universe = pd.read_csv(stock_universe_path)
    stock_universe = stock_universe.values.tolist()
    
    for pairs in stock_universe:
        print(type(pairs[0]))
        Johansen_Strategy(pairs[0])
            
    #Johansen_Strategy(['GLD2','GDX2', 'USO2'])
    #Johansen_Strategy(['USO2','GDX2', 'GLD2'])
    #Johansen_Strategy(['GDX2','GLD2', 'USO2'])
    
if __name__ == '__main__':

        main()