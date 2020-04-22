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
from itertools import combinations
import threading

import talib
from talib import MA_Type

# For drafting email
import win32com.client as win32
from win32com.client import Dispatch, constants

import time

import matplotlib.pyplot as plt
start_date = '2015-01-01'
end_date = datetime.datetime.today().strftime("%Y-%m-%d")

stock_universe_path= r'C:\Users\Evian Zhou\Documents\Python\Pairs Trading\Paris.csv'
stock_hist_data_path = r'C:\Users\Evian Zhou\Documents\Python\Pairs Trading\Hist_data' + '\\'
stock_output_path = r'C:\Users\Evian Zhou\Documents\Python\Pairs Trading\Out_put' + '\\' + end_date + '\\'


# Create folder for output
try:
    os.mkdir(stock_output_path)
except OSError:
    print ("Creation of the directory %s failed" % stock_output_path)
else:
    print ("Successfully created the directory %s " % stock_output_path)



# Define the rolling mean duration
rolling_mean_duration = 420

def listToString(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
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


def variance_calculator(series,series_average,win_len):
	sma = win_len
	temp = series.subtract(series_average)
	temp2 = temp.apply(lambda x: x**2)
	temp3 = temp2.rolling(sma).mean()
	sigma = temp3.apply(lambda x : math.sqrt(x))
	return sigma


def calculate_correlation(Stock_1_file, Stock_2_file, Source_File_Folder):
    
    # Load source file
    stock_1_Close_price = pd.read_csv(Source_File_Folder + Stock_1_file)
    stock_2_Close_price = pd.read_csv(Source_File_Folder + Stock_2_file)
    
    # Extract stock names from source file name
    stock_1_name = Stock_1_file.split('_')[0]
    stock_2_name = Stock_2_file.split('_')[0]
    
    # Process stock price dataframe
    stock_1_Close_price.set_index('Date', inplace=True)
    stock_2_Close_price.set_index('Date', inplace=True)
    
    stock_1_Close_price = stock_1_Close_price['Adj Close']
    stock_2_Close_price = stock_2_Close_price['Adj Close']
    
    # Comine two DataFrame
    pairs = pd.concat([stock_1_Close_price, stock_2_Close_price], axis=1)
    pairs.reset_index(inplace=True)
    pairs.columns = ['Date', stock_1_name + '_Close',stock_2_name + '_Close']
    pairs.set_index('Date', inplace=True)
    
    # Drop NA column for those unmatched date
    pairs.dropna(inplace=True)
    
    # Calculate the ratio
    pairs['Ratio'] = pairs[stock_1_name + '_Close'] / pairs[stock_2_name + '_Close']
    
    # Calculate the rolling mean of the ratio
    pairs['Ratio_Moving_Avg'] = pairs['Ratio'].rolling(rolling_mean_duration).mean()
    
    # Calculate the sigma and bands
    sigma = variance_calculator(pairs['Ratio'], pairs['Ratio_Moving_Avg'], rolling_mean_duration)
    pairs['Sigma'] = sigma
    pairs['Upper_band'] = pairs['Ratio_Moving_Avg'].add(pairs['Sigma'])
    pairs['Lower_band'] = pairs['Ratio_Moving_Avg'].subtract(pairs['Sigma'])
    
    # Check if break
    pairs.loc[(pairs['Ratio'] > pairs['Upper_band']), "Break_Upper"] = True
    pairs.loc[(pairs['Ratio'] < pairs['Lower_band']),  "Break_Lower"] = True
    #print(pairs)
    
    fig, axs = plt.subplots(2, 1, constrained_layout=True,figsize=(16, 12))
    pairs['Ratio'].plot(ax=axs[0], lw=2, label='Ratio')
    pairs['Ratio_Moving_Avg'].plot(ax=axs[0], lw=1,style='--', label='Ratio Moving Avg')
    pairs['Upper_band'].plot(ax=axs[0], lw= 1,style='--', label='Upper Band')
    pairs['Lower_band'].plot(ax=axs[0], lw= 1,style='--', label='Lower Band')
    axs[0].set_xlabel('Date')
    axs[0].set_title('Ratios Moving Avg')
    axs[0].set_ylabel('Ratio')
    axs[0].grid()
    axs[0].legend(loc='upper left')
    
    pairs[stock_1_name + '_Close'].plot(ax=axs[1], lw=1, label=stock_1_name + '_Close')
    pairs[stock_2_name + '_Close'].plot(ax=axs[1], lw=1, label=stock_2_name + '_Close')
    axs[1].set_xlabel('Date')
    axs[1].set_title('Close Price')
    axs[1].set_ylabel('Close Price')
    axs[1].grid()
    axs[1].legend(loc='upper left')
    
    fig.suptitle(stock_1_name + ' VS ' + stock_2_name + ' ' + end_date, fontsize=16)
    fig.savefig(stock_output_path + stock_1_name + ' VS ' + stock_2_name + ' ' + end_date + '.png')
    

    pairs.to_csv(stock_output_path + stock_1_name + '_' + stock_2_name + '_' + end_date + '.csv')

    
    
def loop_thru_files_get_signal(target_date, output_df):
    
    target_sources = os.listdir(stock_output_path)
    target_sources = [s for s in target_sources if target_date in s and '.csv' in s and 'output' not in s]
    
    print("Looping thru all the pairs files and searching for signals...")
    
    for filename in target_sources: 
        filepath = stock_output_path + filename
        
        # Get pairs name from file name
        pairs_name = filename.split('_')[0] + ' ' + filename.split('_')[1]
        
        pairs = pd.read_csv(filepath)
        

        # ONLY Check for the lastest record
        pairs = pairs.iloc[[-1]]
        
        # Add column for pair name
        pairs['Pairs_Name'] = pairs_name
        
        # Filter Break_Upper = true or Break_Lower = True
        pairs = pairs[ (pairs['Break_Upper'].notna()) | (pairs['Break_Lower'].notna()) ]
        pairs.rename(columns={pairs.columns[1]: "Stock_1_Close", pairs.columns[2]: "Stock_2_Close"}, inplace = True)
        # Add the result to output dataframe
        output_df = output_df.sort_values(by=['Pairs_Name'])
        
        
        output_df = pd.concat([output_df, pairs], axis=0)
    
    
    print("Search signals completed!")
    
    return output_df


def loop_thru_combinations(stock_universe_path):
    stock_universe = pd.read_csv(stock_universe_path)
    stock_universe = stock_universe.values.tolist()
    
    print("Looping thru all the combinations and generate pairs..")
    
    for i in stock_universe:
        combination_stock_list = i[0].split(',')
        
        comb = combinations(combination_stock_list, 2) 
  
        # Print the obtained combinations 
        for i in list(comb):
            stock_name_1_path = i[0] + '_' + end_date + '.csv'
            stock_name_2_path = i[1] + '_' + end_date + '.csv'
            
            calculate_correlation(stock_name_1_path, stock_name_2_path, stock_hist_data_path)
    
    print("Pairs data generated!")
            

def draft_email(subject, recipients, dataframe, auto=False): 
    
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)


    if hasattr(recipients, 'strip'):
        recipients = [recipients]

    for recipient in recipients:
        mail.Recipients.Add(recipient)
        
    # Round the decimal
    dataframe = dataframe.round(2)
    
    # Fill NA
    dataframe.fillna('',inplace=True)

    email_body = """
    Hi all,<br>
    Please see the pairs ratios breaking out triggers below:

    """

    email_body = email_body + dataframe.to_html()
    mail.Subject = subject
    mail.HtmlBody = email_body
    
    # Add graphs
    
    for pairs in dataframe['Pairs_Name']:
        pairs = pairs.replace(' ', ' VS ')
        pairs_path = stock_output_path + pairs + ' ' + end_date + '.png'
        
        mail.Attachments.Add(Source = pairs_path)
        
    
    if auto:
        mail.send
    else:
        mail.Display()     
    

def main():
    
    output_df = pd.DataFrame(columns=['Date', 'Pairs_Name', 'Stock_1_Close', 'Stock_2_Close', 'Ratio', \
                                               'Ratio_Moving_Avg', 'Sigma', 'Upper_band', 'Lower_band', \
                                                   'Break_Upper', 'Break_Lower'])
    
    Download_stock_price(stock_universe_path, stock_hist_data_path)

    loop_thru_combinations(stock_universe_path)

    output_df = loop_thru_files_get_signal(end_date, output_df)
    
    
    # Save output file
    output_df.to_csv(stock_output_path +'output_' + end_date + '.csv')
    

    # Draft email
    draft_email('Paris ratio ' + end_date, 'investment@optimascap.com', output_df,  auto=False)
    
    
if __name__ == '__main__':

        main()