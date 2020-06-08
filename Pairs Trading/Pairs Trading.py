
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 23:42:41 2020

@author: Evian Zhou
"""

# Pandas and Numpy for data manipulation and Matplotlib for ploting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Math related library
import math
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from johansen import coint_johansen
from statsmodels.regression.rolling import RollingOLS
from hurst import compute_Hc

# Yahoo Finance for download the stock OHLC prices
import yfinance as yf

# Basic system function
import datetime
import time
import os

# For solving the combinations and create pairs
from itertools import combinations

# Ta-lib python library for technical analysis
import talib
from talib import MA_Type

# For drafting email
import win32com.client as win32
from win32com.client import Dispatch, constants


# Define the dates and file location
start_date = '2015-01-01'
end_date = datetime.datetime.today().strftime("%Y-%m-%d")

stock_universe_path = r'U:\Python\Pairs Trading\Pairs.csv'
stock_hist_data_path = r'U:\Python Trading Output\Stock_OHLC' + '\\' + end_date + '\\'
stock_output_path = r'U:\Python Trading Output\Pairs Trading\Out_put' + '\\' + end_date + '\\'

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



CADF_Result_list = pd.DataFrame(columns=['Pairs_Name', 'CADF_result'])

# Define the rolling mean duration
rolling_mean_duration = 210
hedge_ratio_duration = 210
hedge_ratio_rolling_duration = 210

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

def hedge_ratio_calculator(df_1, df_2):
    model = sm.OLS(df_1, df_1)
    model = model.fit() 
    print('Hedge Ratio =', model.params[0])
    
    return model.params[0]
    

def calculate_correlation(Stock_1_file, Stock_2_file, Source_File_Folder):
    global CADF_Result_list
    
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
    
    # Drop Duplicates:
    #stock_1_Close_price = stock_1_Close_price[stock_1_Close_price.drop_duplicates(keep='last')]
    #stock_1_Close_price = stock_2_Close_price[stock_2_Close_price.drop_duplicates(keep='last')]
    #stock_1_Close_price.drop_duplicates(keep='last', inplace = True)
    #stock_2_Close_price.drop_duplicates(keep='last', inplace = True)
    
    print(stock_2_Close_price.tail())
    
    # Comine two DataFrame
    print("Concating Dataframe for %s and %s" %(stock_1_name, stock_2_name ))
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
    
    
    print(pairs)
    
    
    # Calculate the hedge ratio
    model = sm.OLS(pairs.iloc[-hedge_ratio_duration:, :1], pairs.iloc[-hedge_ratio_duration:, 1:2])
    model = model.fit() 
    print('Hedge Ratio =', model.params[0])
    pairs['Hedge_Ratio'] = model.params[0]
    #pairs['Half_Life'] =  -np.log(2) / model.params[0]
    
    print("*****************************")
    print(-np.log(2) / model.params[0])
    print("*****************************")
    
    #fit
    model_rolling = RollingOLS(endog=pairs.iloc[:, :1] , exog=pairs.iloc[:, 1:2],window=hedge_ratio_rolling_duration)
    rres = model_rolling.fit()
    rres_result = rres.params
    #rres.params.tail() #get intercept and coef
    print("***")
    rres_result.rename(columns={rres_result.columns[0]: "Hedge_Ratio_Rolling"}, inplace = True)
    print(rres_result)
    print("***")
    pairs = pd.concat([pairs, rres_result], axis=1)
    pairs['Hedge_Ratio_Rolling'].fillna(method='bfill', inplace=True)
    
    
    # Calculate the spread MA and band
    #pairs['Spread'] = pairs[stock_1_name + '_Close'] - model.params[0] * pairs[stock_2_name + '_Close']
    pairs['Spread'] = pairs[stock_1_name + '_Close'] - pairs['Hedge_Ratio_Rolling'] * pairs[stock_2_name + '_Close']
    
    pairs['Spread_Moving_Avg'] = pairs['Spread'].rolling(rolling_mean_duration).mean()
    Spread_Sigma = variance_calculator(pairs['Spread'], pairs['Spread_Moving_Avg'], rolling_mean_duration)
    pairs['Spread_Sigma'] = Spread_Sigma
    pairs['Spread_Upper_band'] = pairs['Spread_Moving_Avg'].add(pairs['Spread_Sigma'])
    pairs['Spread_Lower_band'] = pairs['Spread_Moving_Avg'].subtract(pairs['Spread_Sigma'])
    
    # Half_Life
    print("*** Half Life ***")
    pairs_spread = pairs['Spread'].iloc[-hedge_ratio_duration:,]
    
    # Spread and differenence between spread
    spread_x = np.mean(pairs_spread) - pairs_spread 
    spread_y = pairs_spread.shift(-1) - pairs_spread
    
    spread_df = pd.DataFrame({'x':spread_x,'y':spread_y})
    spread_df = spread_df.dropna()
    
    # Theta as regression beta between spread and difference between spread
    model_s = sm.OLS(spread_df['y'], spread_df['x'])
    model_s = model_s.fit() 
    theta=  model_s.params[0]
    # Type your code below
    Half_Life = math.log(2)/theta
    Half_Life = round(Half_Life, 2)
    print(Half_Life)
    pairs['Half_Life'] = Half_Life
    
    # Hurst Exponent
    H, c, data = compute_Hc(pairs_spread)
    print("*** Hurst exponent ***")
    H = round(H, 2)
    print(H)
    pairs['Hurst_Exponent'] = H
    
    # CADF Stationarity test
    CADF_result = ts.adfuller(pairs['Spread'])
    CADF_Result_list = CADF_Result_list.append({'Pairs_Name' : stock_1_name + ' ' + stock_2_name , 'CADF_result' : CADF_result[0], 'Half_Life' : Half_Life, 'Hurst_Exponent': H} , ignore_index=True)
    print('CAFD_result for ' + stock_1_name + ' ' + stock_2_name + ' : %.3f' % CADF_result[0])
    print('CAFD Critial values:')
    for key, value in CADF_result[4].items():
        print('\t%s: %.3f' % (key, value))
        
        
    # Johansen Stationarity test
    result = coint_johansen(pairs.iloc[:,:2] ,0 ,1)
    d = result.evec
    ev= d[0]

    # Normalizing the eigenvectors
    ev = ev/ev[0]
    print(ev[0])
    
    # Create graph
    print("Creating Plot for %s and %s" %(stock_1_name, stock_2_name ))
    
    fig, axs = plt.subplots(3, 1, constrained_layout=True,figsize=(16, 12))
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
    
    
    pairs['Spread'].plot(ax=axs[2], lw=2, label='Spread')
    pairs['Spread_Moving_Avg'].plot(ax=axs[2], lw=1,style='--', label='Spread Moving Avg')
    pairs['Spread_Upper_band'].plot(ax=axs[2], lw= 1,style='--', label='Spread Upper Band')
    pairs['Spread_Lower_band'].plot(ax=axs[2], lw= 1,style='--', label='Spread Lower Band')
    axs[2].set_xlabel('Date')
    axs[2].set_title('Spread Moving Avg')
    axs[2].set_ylabel('Spread')
    axs[2].grid()
    axs[2].legend(loc='upper left')
    
    fig.suptitle(stock_1_name + ' VS ' + stock_2_name + ' ' + end_date, fontsize=16)
    fig.savefig(stock_output_path + stock_1_name + ' VS ' + stock_2_name + ' ' + end_date + '.png')
    

    pairs.to_csv(stock_output_path + stock_1_name + '_' + stock_2_name + '_' + end_date + '.csv')

    
    
def loop_thru_files_get_signal(target_date, output_df):
    
    target_sources = os.listdir(stock_output_path)
    target_sources = [s for s in target_sources if target_date in s and '.csv' in s and 'output' not in s and 'CADF' not in s]
    
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

    
    dataframe = dataframe[['Date', 'Pairs_Name','Stock_1_Close', 'Stock_2_Close', 'Ratio', 'Ratio_Moving_Avg', 'Sigma', \
                          'Upper_band','Lower_band', 'Break_Upper', 'Break_Lower','Half_Life']]

    
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
    global CADF_Result_list
    
    output_df = pd.DataFrame(columns=['Date', 'Pairs_Name', 'Stock_1_Close', 'Stock_2_Close', 'Ratio', \
                                               'Ratio_Moving_Avg', 'Sigma', 'Upper_band', 'Lower_band', \
                                                   'Break_Upper', 'Break_Lower', 'Half_life'])
    
    Download_stock_price(stock_universe_path, stock_hist_data_path)

    loop_thru_combinations(stock_universe_path)

    output_df = loop_thru_files_get_signal(end_date, output_df)
    
    # Save output file
    output_df.to_csv(stock_output_path +'output_' + end_date + '.csv')
    CADF_Result_list.to_csv(stock_output_path +'CADF_Result_list' + end_date + '.csv')
    
    # Draft email
    draft_email('Pairs ratio ' + end_date, 'investment@optimascap.com,stephen.tong@hld.com', output_df,  auto=False)
    
    
    print(CADF_Result_list)
    
if __name__ == '__main__':

        main()