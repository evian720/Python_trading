# coding=utf-8
import datetime as dt
import os
import pytz

from configuration import test_me
from data_provider_factory.data_loading_plan import HistIngestionPlan, Plan
from IBridgePy.IbridgepyTools import symbol, superSymbol

# Related YouTube tutorials about IBridgePy backtester
# Build “Buy low Sell high” strategy by Machine learning https://youtu.be/hNCwNxeXrwA
# Detailed explanation about “Buy low Sell high” strategy https://youtu.be/PI5dhqCAuvA
# Calculate Sharpe Ratio https://youtu.be/4xTHdzAMhcI
# Backtest without hist ingestion https://youtu.be/bwmx5hiSPV4
# Backtest with hist ingestion https://youtu.be/XnpxAVU4ogY
# Backtest with hist from local files https://youtu.be/UR_7_F8wPL0
# Speed up backtest by designating spot times https://youtu.be/bVE59nZ02ig
# Convert hist data format https://youtu.be/hYL6SYgy7wE
fileName = 'demo_buy_low_sell_high.py'

accountCode = 'dummyAccountCode'  # IBridgePy needs a dummyAccountCode to simulate transactions when dataProviderName = 'LOCAL_FILE'
dataProviderName = 'LOCAL_FILE'  # RANDOM, IB, LOCAL_FILE, TD, ROBINHOOD, IBRIDGEPY

# "histIngestionPlan" is a reserved word in IBridgePy to store the historical data ingestion plan that describes what historical data
# are needed during backtesting and IBridgePy backtester will fetch these data before backtesting to speed up the whole backtesting process.
# "histIngestionPlan" is not required for backtesting but it will make backtest much faster.
# "histIngestionPlan" is an instance of HistIngestionPlan.
histIngestionPlan = HistIngestionPlan()
histIngestionPlan.add(Plan(security=symbol('SPY'), barSize='1 min', fileName='STK,SPY,USD_1 min_40 D.csv'))  # "histIngestionPlan.add" is used to add more Ingestion Plans
histIngestionPlan.add(Plan(security=symbol('SPY'), barSize='1 day', fileName='STK,SPY,USD_1 day_55 D.csv'))  # "histIngestionPlan.add" is used to add more Ingestion Plans

# In CUSTOM mode, user can specify each spot time to backtest with any data provider.
# customSpotTimeList is an IBridgePy reserved key word. Its type is List.
# User can append each spot time into customSpotTimeList to backtest at these spot times.
timeGeneratorType = 'CUSTOM'
customSpotTimeList = []
import pandas as pd
# The goal is to backtest from 2020-10-30 to 2020-12-24 and only allow the datetime of 15:59:00 Eastern go through the to-be-tested strategy
# because "demo_close_price_reversion.py" only makes trading decisions at 15:59:00 Eastern on every trading day.
# pd.date_range is used to create a list of dates that have hour=0 and minute=0.
dateRange = pd.date_range(dt.datetime(2020, 10, 30), dt.datetime(2020, 12, 24), freq='1D', tz=pytz.timezone('US/Eastern'))
for aDate in dateRange:
    a = aDate.replace(hour=15, minute=59)  # datetime(2020, 10, 30, 0, 0) --> datetime(2020, 10, 30, 15, 59)
    customSpotTimeList.append(a)

test_me(fileName, globals())

# Related YouTube tutorials about IBridgePy backtester
# Build “Buy low Sell high” strategy by Machine learning https://youtu.be/hNCwNxeXrwA
# Detailed explanation about “Buy low Sell high” strategy https://youtu.be/PI5dhqCAuvA
# Calculate Sharpe Ratio https://youtu.be/4xTHdzAMhcI
# Backtest without hist ingestion https://youtu.be/bwmx5hiSPV4
# Backtest with hist ingestion https://youtu.be/XnpxAVU4ogY
# Backtest with hist from local files https://youtu.be/UR_7_F8wPL0
# Speed up backtest by designating spot times https://youtu.be/bVE59nZ02ig
# Convert hist data format https://youtu.be/hYL6SYgy7wE
