import numpy as np
import pandas as pd
from stocks import StockAnalysis
from datetime import datetime
import matplotlib.pyplot as plt

total_invest = 100000
number_of_investments = int(52)
invest_on_days = [4, 11, 18, 25]
# invest_on_days = [1]
stock = 'FLCEX'
buy_at = 'Close' # Open or Close
startdate = datetime(2000,1,1)
enddate = datetime(2021,7,6)
periodic_investment = round(total_invest / number_of_investments, 2)
total_invest = periodic_investment * number_of_investments

if False:
    df = StockAnalysis(stock, startdate, enddate).df
else:
    df = pd.read_csv('data/FLCEX.csv', parse_dates=['Date'])
    df = df.reset_index(drop=True)
    df = df.sort_values('Date')
    df = df[~df.Date.duplicated(keep='last')]
    df = df.set_index('Date')

# set index to Date, resample dates to fill missing days, backfill nans
df = df.resample('D').first().fillna(method='backfill')
df['Date'] = df.index
df = df[['Date', buy_at]]
df = df.rename(columns={buy_at: stock})
buy_at = stock
df.reset_index(drop=True, inplace=True)
df['invest'] = 0
df['own'] = 0
for day in invest_on_days:
    df.loc[df.Date.dt.day == day, 'invest'] = periodic_investment
    df.loc[df.Date.dt.day == day, 'own'] = periodic_investment / df[buy_at][df.Date.dt.day == day]
df = df[df.invest != 0]
df['ls' ] = total_invest / df[buy_at]
df['dca'] = 0
for i in range(len(df) + 1 - number_of_investments):
    df.iloc[i, -1] = df.iloc[i:i + number_of_investments].own.cumsum().iloc[-1]

df['dca_aggressive'] = 0
for i in range(len(df) + 1 - number_of_investments):
    df1 = df.iloc[i:i + number_of_investments].copy()
    base_price = df1[buy_at].iloc[0]
    df1.invest = periodic_investment * (1 + 1*(1 - df1[buy_at] / base_price) )
    df1.loc[df1.invest < periodic_investment, 'invest'] = periodic_investment
    df1['total'] = df1.invest.cumsum()
    remaining = total_invest - df1[df1.total < total_invest].iloc[-1].total
    remaining_index = df1[df1.total >= total_invest].index[0]
    df1.loc[remaining_index, 'invest'] = remaining
    df1.own = df1.invest / df1[buy_at]
    df1['total'] = df1.invest.cumsum()
    df1 = df1[df1.total <= total_invest]
    df.iloc[i, -1] = df1.own.cumsum().iloc[-1]

df['percent_more_dca'] = (df.dca / df.ls - 1) * 100
df['percent_more_dca_aggressive'] = (df.dca_aggressive / df.ls - 1) * 100
df.loc[df.percent_more_dca == -100, 'percent_more_dca'] = np.nan
df.loc[df.percent_more_dca_aggressive == -100, 'percent_more_dca_aggressive'] = np.nan
# df = df[df.dca > 0]

df.ls = df.ls.shift(number_of_investments)
df.dca = df.dca.shift(number_of_investments)
df['ls_worth'] = df.ls * df[buy_at]
df['dca_worth'] = df.dca * df[buy_at]

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
df.plot(ax=ax[0], x='Date', y='percent_more_dca')
df.plot(ax=ax[0], x='Date', y='percent_more_dca_aggressive')
df.plot(ax=ax[0], x='Date', y=buy_at, secondary_y=True)
ax[0].axhline(y=0, color='k', alpha=0.5)

df.plot(ax=ax[1], x='Date', y='ls_worth')
df.plot(ax=ax[1], x='Date', y='dca_worth')
plt.show()
