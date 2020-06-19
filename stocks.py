import argparse
import mplfinance as mpf
import numba as nb
import os
import pandas as pd
from pandas_datareader import data, wb
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
from pandas.tseries.holiday import USFederalHolidayCalendar
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import sys
# %matplotlib inline
plt.rcParams['figure.figsize'] = [15, 15]
plt.style.use('ggplot')
# plt.style.use('seaborn')


from matplotlib.ticker import Formatter
class WeekdayDateFormatter(Formatter):
    # https://matplotlib.org/gallery/ticks_and_spines/date_index_formatter.html
    # the data is first plotted against an integer. The formatter changes the integer to the correct date.
    def __init__(self, dates, fmt='%Y-%m-%d'):
        self.dates = dates
        self.fmt = fmt
    def __call__(self, x, pos=0):
        'Return the label for time x at position pos'
        ind = int(round(x))
        if ind >= len(self.dates) or ind < 0:
            return ''
        return (self.dates[ind]).strftime(self.fmt)


def search_stock_symbols(stock):
    fn = 'stock_symbols.csv'
    if not os.path.exists(fn):
        symbols = get_nasdaq_symbols()
        symbols.to_csv(fn, index='Symbol')
    else:
        symbols = pd.read_csv(fn, index_col='Symbol')
    if stock is None:
        return symbols
    stock = stock.upper()
    hard_search = symbols[symbols['NASDAQ Symbol'] == stock]
    if len(hard_search) == 1:
        return 1, symbols[symbols['NASDAQ Symbol'] == stock]['Security Name'][stock]
    else:
        found = symbols[symbols['NASDAQ Symbol'].str.contains(stock)]
        if found.empty:
            return 0, None
        else:
            return len(found), found



def valid_time(arg):
    try:
        return datetime.strptime(arg, "%H:%M")
    except ValueError:
        msg = "Not a valid time: '{0}'.".format(arg)
        raise argparse.ArgumentTypeError(msg)


def valid_date(arg):
    try:
        dt = datetime.strptime(arg, "%m/%d/%Y")
    except ValueError:
        msg = 'Not a valid date: "{0}".'.format(arg)
        raise argparse.ArgumentTypeError(msg)
    if dt.date() > datetime.now().date():
        dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        msg = f'''A future date is not valid: "{arg}". Instead using "{dt.date().strftime('%m/%d/%Y')}"'''
        print(msg)
    return dt


def cli_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('stock', metavar='N', type=str, nargs='*', help='create reports for all stocks entered')
    parser.add_argument('--compare', action='store_true', default=False, help='compare the list of stocks')
    parser.add_argument('--bb', action='store_true', default=False, help='show Bollinger Bands on stock chart')
    parser.add_argument('--macd', action='store_true', default=False, help='show Moving Average Convergence/Divergence on separate chart')
    parser.add_argument('--sto', action='store_true', default=False, help='show Stochastic on separate chart')
    parser.add_argument('--rsi', action='store_true', default=False, help='show Relative Srength Index on separate chart')
    parser.add_argument('--cmf', action='store_true', default=False, help='show Chaikin Money Flow on separate chart')
    parser.add_argument('--best', action='store_true', default=False, help='show BB, MACD, and RSI')
    parser.add_argument('--save', action='store_true', default=False, help='Save plot to disk')
    parser.add_argument('--show', action='store_true', default=False, help='Show interactive plot')
    parser.add_argument("--start", help="Start date - format MM/DD/YYYY",
        default=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=547), type=valid_date)
    parser.add_argument("--end", help="End date - format MM/DD/YYYY",
        default=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0), type=valid_date)
    args = parser.parse_args()
    args.stock = sorted([i.upper() for i in args.stock])
    if not args.save:
        args.show = True
    # if len(args.stock) > 1:
    #     args.save = True
    #     args.show = False
    if args.start > args.end:
        parser.error(f'Start date "{args.start}" can not be greater than End Date "{args.end}"')
    if args.best:
        args.bb = True
        args.macd = True
        args.rsi = True
    # log_message(parser.parse_args().__str__())
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return args


def remove_holidays_and_weekends(start, end, move_date_forward=True):
    holidays = USFederalHolidayCalendar().holidays(start=start - timedelta(days=14), end=end + timedelta(days=14)).to_pydatetime()
    if move_date_forward:
        dt = start
    else:
        dt = end
    while dt in holidays or dt.weekday() >= 5:
        if move_date_forward:
            dt += timedelta(days=1)
        else:
            dt -= timedelta(days=1)
    return dt


class StockAnalysis:
    def __init__(self, stock: str, start: datetime, end: datetime,
        sma: list=[200, 50, 5], close_col: str="Close", plot_type: str="line"):
        """
        Gather data for the stock between the given dates.
        SMA: list of simple moving average days to plot
        CLOSE_COL: column name to use for close data. Usually Close or Adj Close
        PLOT_TYPE: how to plot the data. line or candlestick
        """
        self.stock = stock
        self.stock_name = None
        self.stock_count = self.confirm_stock_symbol()
        if self.stock_count > 1:sys.exit(1)
        self.start = start
        self.end = end
        self.close_col = close_col
        self.sma = sma
        self.plot_type = plot_type
        self.df = self.get_data_frame(self.stock, self.start, self.end)
        self.set_day_color()
        self.simple_moving_average()


    def confirm_stock_symbol(self):
        count, name = search_stock_symbols(self.stock)
        if count == 0:
            print(f'Symbol {self.stock} is not traded on the Nasdaq exchange')
        elif count == 1:
            self.stock_name = name
        else:
            print(f'Multiple stock symbols found for {self.stock}')
        return count


    def store_stock(self, stock, start, end, filename):
        print(f"Pulling stock data for {stock} from {start} to {end}")
        try:
            df = pd.DataFrame(data.DataReader(stock,'yahoo',start,end))
        except KeyError:
            print("Stock out of range")
            df = pd.DataFrame()
        df = df.reset_index()
        if os.path.exists(filename):
            df_existing = pd.read_csv(filename, parse_dates=['Date'])
            df = df_existing.append(df).reset_index(drop=True)
            df = df.sort_values('Date')
        if df.empty and self.stock_name == 0:
            print(f"No data found for {self.stock}")
            sys.exit(1)
        # sometimes data is returned with two rows for the same date. The last row is the row to keep.
        df = df[~df.Date.duplicated(keep='last')]
        df_store = df.copy()
        market_close = datetime.now().replace(hour=15, minute=5, second=0, microsecond=0)
        # market_close = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
        if (df.Date.sort_values().iloc[-1].date() == datetime.today().date()) and (datetime.now() < market_close):
            # The market has not closed today so do not store today's data in csv.
            df_store.drop(df_store.tail(1).index,inplace=True)
        df_store.to_csv(filename, index=False)
        return df


    def get_data_frame(self, stock, start, end, get_most_recent_data: bool = True):
        """
        :stock: text stock ticker
        :start: date to start stock data in format "MM-DD-YYYY" or python datetime
        :end: date to end stock data in format "MM-DD-YYYY" or python datetime
        # :get_most_recent_data: update stored data to have recent close data
        """
        start_dt = remove_holidays_and_weekends(start, end, move_date_forward=True)
        end_dt = remove_holidays_and_weekends(start, end, move_date_forward=False)
        filename = f"data/{stock}.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename, parse_dates=['Date'])
            if start_dt >= df.Date.min() and end_dt <= df.Date.max():
                print(f"Using Stored Stock Data for {stock} from {start_dt.date()} to {end_dt.date()}")
            if end_dt > df.Date.max():
                interim_dt = remove_holidays_and_weekends(df.Date.max() + pd.Timedelta("1d"), end_dt, move_date_forward=True)
                if interim_dt <= end_dt and interim_dt > df.Date.max():
                    df = self.store_stock(stock, interim_dt, end_dt, filename)
            if start_dt < df.Date.min():
                interim_dt = remove_holidays_and_weekends(start_dt, df.Date.min() - pd.Timedelta("1d"), move_date_forward=False)
                if interim_dt >= start_dt and interim_dt < df.Date.min():
                    df = self.store_stock(stock, start_dt, interim_dt, filename)
        else:
            df = self.store_stock(stock, start_dt, end_dt, filename)
        df.Date = pd.to_datetime(df.Date)
        df = df[(df.Date >= start_dt) & (df.Date <= end_dt)].copy()
        df.set_index('Date', inplace=True)
        df['TOpen'] = df.Open.shift(-1) # tomorrow's open
        # df = df.asfreq('D')
        return df


    def simple_moving_average(self):
        for sma in self.sma:
            self.df[f'{sma} mavg'] = self.df[self.close_col].rolling(window=sma, min_periods=sma).mean()


    def set_day_color(self):
        self.df['day_color'] = 'red'
        self.df.loc[self.df[self.close_col] >= self.df.Open, 'day_color'] = 'green'
        self.df.loc[self.df[self.close_col] == self.df.Open, 'day_color'] = 'gray'


    def MACD(self, big_ema: int = 26, small_ema: int = 12, signal_span: int = 9):
        """
        TREND INDICATOR
        Moving Average Convergence/Divergence
        """
        # period = 200
        # sma = self.df[f'{period} mavg'][:period]
        # rest_of_close = self.df[self.close_col][period:]
        # series_for_ewm = pd.concat([sma, rest_of_close]) 
        series_for_ewm = self.df[self.close_col]
        self.df[f'{big_ema} ema'] = series_for_ewm.ewm(span=big_ema, adjust=False).mean()
        self.df[f'{small_ema} ema'] = series_for_ewm.ewm(span=small_ema, adjust=False).mean()
        self.df['MACD'] = (self.df[f'{small_ema} ema'] - self.df[f'{big_ema} ema'])
        self.df['Signal'] = self.df['MACD'].ewm(span=signal_span, adjust=False).mean()
        self.df['Crossover'] = self.df['MACD'] - self.df['Signal']
        self.df['YCrossover'] = self.df.Crossover.shift() #yesterday crossover
        self.df['MACD_indicator'] = 0
        self.df.loc[(self.df.Crossover < 0) & (self.df.YCrossover > 0), 'MACD_indicator'] = 1 # Sell, cross line going negative
        self.df.loc[(self.df.Crossover > 0) & (self.df.YCrossover < 0), 'MACD_indicator'] = 2 # Buy, cross line going positive


    def Bollinger_Bands(self, n: int = 20, ndev: float = 2.0):
        """
        VOLATILITY INDICATOR
        Bollinger Bands
        """
        self.bb_mavg = f'{n} mavg'
        self.df[self.bb_mavg] = self.df[self.close_col].rolling(window=n).mean()
        # set .std(ddof=0) for population standard deviation instead of sample deviation
        self.df['BBstd'] = self.df[self.close_col].rolling(window=n).std(ddof=0)
        self.df['BBUpper'] = self.df[self.bb_mavg] + self.df.BBstd * ndev
        self.df['BBLower'] = self.df[self.bb_mavg] - self.df.BBstd * ndev
        self.df['BB_indicator'] = 0
        self.df.loc[(self.df[self.close_col] < self.df['BBLower']), 'BB_indicator'] = 1 # close was below band
        self.df.loc[(self.df[self.close_col] > self.df['BBUpper']), 'BB_indicator'] = 2 # close was above band


    def Stochastic(self, n: int=14, d_n: int=3, over: float=80, under: float=20):
        """
        MOMENTUM INDICATOR
        Stochastic Oscillator Indicator
        STOL = Stochastic Low
        STOH = Stochastic High
        STOK = Stochastic %K
        STOD = Stochastic %D
        n = number of days to consider
        d_n = rolling average for D line
        over = above this percent value is over bought territory
        under = below this percent value is over sold territory
        """
        self.sto_over = over
        self.sto_under = under
        self.df['STOL'] = self.df['Low'].rolling(window=n).min()
        self.df['STOH'] = self.df['High'].rolling(window=n).max()
        self.df['STOK'] = 100 * ( (self.df[self.close_col] - self.df.STOL) / (self.df.STOH - self.df.STOL) ) # fast
        self.df['STOD'] = self.df['STOK'].rolling(window=d_n).mean() # slow
        self.df['STO_indicator'] = 0
        self.df.loc[(self.df.STOK < self.df.STOD) & 
            (self.df.STOK.shift(1) > self.df.STOD.shift(1)) & 
            (self.df.STOD > over), 'STO_indicator'] = 1 # Sell, fast crosses below slow in the high range
        self.df.loc[(self.df.STOK > self.df.STOD) & 
            (self.df.STOK.shift(1) < self.df.STOD.shift(1)) & 
            (self.df.STOD < under), 'STO_indicator'] = 2 # Buy, fast crosses up over slow in the low range


    # @nb.jit(fastmath=True, nopython=True)
    def RSI(self, n: int = 14, over: float=70, under: float=30):
        """
        MOMENTUM INDICATOR
        Relative Strength Index
        over = above this percent line is over bought territory
        under = below this percent line is over sold territory

        """
        self.rsi_over = over
        self.rsi_under = under
        self.df['RSIchange'] = self.df[self.close_col].diff(1)
        self.df['RSIgain'] = 0
        self.df['RSIloss'] = 0
        self.df.loc[self.df.RSIchange > 0, 'RSIgain'] = self.df.RSIchange
        self.df.loc[self.df.RSIchange < 0, 'RSIloss'] = -self.df.RSIchange
        self.df['AvgGain'] = self.df.RSIgain.ewm(com=n - 1, min_periods=n, adjust=False).mean()
        self.df['AvgLoss'] = self.df.RSIloss.ewm(com=n - 1, min_periods=n, adjust=False).mean()
        self.df['RSI'] = 100 - (100 / (1 + abs(self.df.AvgGain / self.df.AvgLoss)))
        self.df['RSI_indicator'] = 0
        self.df.loc[self.df.RSI > over, 'RSI_indicator'] = 1 # Sell, in overbought range
        self.df.loc[self.df.RSI < under, 'RSI_indicator'] = 2 # Buy, in oversold range


    def CMF(self, n: int = 20, buffer: float=0.05):
        """
        VOLUME INDICATOR
        Chaikin Money Flow Indicator 
        Money Flow Multiplier = ((Close value – Low value) – (High value – Close value)) / (High value – Low value)
        Money Flow Volume = Money Flow Multiplier x Volume for the Period
        CMF = n-day Average of the Daily Money Flow Volume / n-day Average of the Volume
        buffer = above this buffer is bullish buy territory
                 below this negative buffer is bearish sell territory
        """
        self.df["MFV"] = ((self.df[self.close_col] - self.df.Low) - 
                (self.df.High - self.df[self.close_col])) / (self.df.High - self.df.Low) * self.df.Volume
        self.df["CMF"] = self.df.MFV.rolling(window=n, min_periods=n).mean() / self.df.Volume.rolling(window=n, min_periods=n).mean()
        self.df["CMF_indicator"] = 0
        self.df.loc[self.df.CMF < -buffer, "CMF_indicator"] = 1 # Sell, crossed into negative territory 
        self.df.loc[self.df.CMF > buffer, "CMF_indicator"] = 2 # Buy, crossed into positive territory 
        self.cmf_buffer = buffer


    def plot_data_mpf(self):
        mc = mpf.make_marketcolors(up='g', down='r', edge='inherit', wick='inherit', volume='inherit', ohlc='inherit', alpha=0.5) #style="charles"
        s  = mpf.make_mpf_style(base_mpl_style='seaborn', marketcolors=mc)
        mpf.plot(self.df, type='candlestick', mav=(1, 200), volume=True, style=s, figratio=(16,8), figscale=1, title=self.stock)


    def candlestick_plot(self, ax=None, positive_color: str='g', negative_color: str='r'):
        width_bar = 0.8
        width_stick = 0.15
        self.df['bar_top'] = self.df.Open
        self.df.loc[self.df[self.close_col] >= self.df.Open, 'bar_top'] = self.df[self.close_col]
        self.df['bar_bot'] = self.df.Open
        self.df.loc[self.df[self.close_col] < self.df.Open, 'bar_bot'] = self.df[self.close_col]
        ax.bar(x=self.df.index, height=self.df.bar_top - self.df.bar_bot, width=width_bar, bottom=self.df.bar_bot, color=self.df.day_color, edgecolor=self.df.day_color, alpha=0.5)
        ax.bar(x=self.df.index, height=self.df.High - self.df.bar_top, width=width_stick, bottom=self.df.bar_top, color=self.df.day_color, edgecolor=self.df.day_color, alpha=0.5)
        ax.bar(x=self.df.index, height=self.df.Low - self.df.bar_bot, width=width_stick, bottom=self.df.bar_bot, color=self.df.day_color, edgecolor=self.df.day_color, alpha=0.5)
        return ax


    def plot_data(self, show_plot: bool = False, save_plot: bool = True):
        # make Date a column in the DataFrame
        self.df.reset_index(inplace=True)
        # number of charts to create is determined by finding columns in the dataframe
        extra_charts = len(set(["MACD", "STOK", "CMF", "RSI"]).intersection(set(self.df.columns)))
        fig, axs = plt.subplots(2 + extra_charts, 1, sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': [7, 1] + [2] * extra_charts})
        axs_count = 0
        text_offset = 1
        text_lower = 0.05 * self.df[self.close_col].iloc[-1]


        self.df.plot(y=self.close_col, ax=axs[0])
        if self.plot_type == 'candlestick':
            axs[0] = self.candlestick_plot(ax=axs[0])
        else:
            self.df.plot(y=self.close_col, ax=axs[0])
        for sma in self.sma:
            if self.df[f'{sma} mavg'].count() > 0:
                self.df.plot(y=f'{sma} mavg', ax=axs[0], linestyle='--')
        axs_count += 1
        axs[axs_count].bar(x=self.df.index, height=self.df.Volume, width=1, color=self.df.day_color)
        # self.df.plot(y='Volume', kind='bar', ax=axs[axs_count])
        # axs0 = axs[0].twinx()

        if "BBstd" in self.df.columns:
            axs[0].fill_between(self.df.index, self.df.BBUpper, self.df.BBLower, color='gray', alpha=0.3)
            self.df.plot(y=self.bb_mavg, ax=axs[0], linestyle="--")


        if "MACD" in self.df.columns:
            self.df.index.name = 'orig_index'
            self.df.reset_index(inplace=True)
            bull_bear_period_transparency = 0.2
            bullish_momentum_periods = self.df[(self.df.MACD > 0) & (self.df.MACD > self.df.Signal)]['orig_index']
            for row, dfg in bullish_momentum_periods.groupby((bullish_momentum_periods.diff() != 1).cumsum()):
                axs[0].axvspan(dfg.index.min(), dfg.index.max(), color='green', alpha=bull_bear_period_transparency)
                
            # bearish_momentum_periods = self.df[(self.df.MACD < 0) & (self.df.MACD < self.df.Signal)]['Date']
            # for row, dfg in bearish_momentum_periods.groupby((bearish_momentum_periods.diff() != pd.Timedelta('1d')).cumsum()):
            #     axs[0].axvspan(dfg.index.min(), dfg.index.max(), color='red', alpha=bull_bear_period_transparency)

            bearish_momentum_periods = self.df[(self.df.MACD < 0) & (self.df.MACD < self.df.Signal)]['orig_index']
            for row, dfg in bearish_momentum_periods.groupby((bearish_momentum_periods.diff() != 1).cumsum()):
                axs[0].axvspan(dfg.index.min(), dfg.index.max(), color='red', alpha=bull_bear_period_transparency)

            axs_count += 1
            # MACD buy sell indicators
            for index, row in self.df[self.df.MACD_indicator == 2].iterrows():
                axs[axs_count].text(index, row.MACD, 'B', color='g')
            for index, row in self.df[self.df.MACD_indicator == 1].iterrows():
                axs[axs_count].text(index, row.MACD, 'S', color='r')
            # MACD bars
            self.df["MACD Crossover diff"] = self.df.Crossover.diff(1)
            self.df["MACD bar color"] = 'r'
            self.df.loc[self.df["MACD Crossover diff"] > 0, "MACD bar color"] = 'g'
            axs[axs_count].bar(self.df.index, self.df.Crossover, width=1, color=self.df["MACD bar color"])
            axs[axs_count].axhline(y=0, color='gray', linestyle='-.')
            self.df.plot(y=['MACD', 'Signal', 'Crossover'], ax=axs[axs_count])
            axs[axs_count].legend(loc='center left')


        if 'STOK' in self.df.columns:
            axs_count += 1
            axs[axs_count].axhline(y=self.sto_over, color='k', linestyle=':')
            axs[axs_count].axhline(y=self.sto_under, color='k', linestyle=':')
            axs[axs_count].fill_between(self.df.index, self.sto_over, 
                self.df.STOK, where=self.df.STOK > self.sto_over,
                interpolate=True, color='red', alpha=0.5)
            axs[axs_count].fill_between(self.df.index, self.sto_under, 
                self.df.STOK, where=self.df.STOK < self.sto_under,
                interpolate=True, color='green', alpha=0.5)
            self.df.plot(y=['STOK', 'STOD'], ax=axs[axs_count])
            axs[axs_count].legend(loc='center left')

            for index, row in self.df[self.df.STO_indicator == 2].iterrows():
                axs[axs_count].text(index, row.STOK, 'B', color='g')
            for index, row in self.df[self.df.STO_indicator == 1].iterrows():
                axs[axs_count].text(index, row.STOK, 'S', color='r')


        if 'CMF' in self.df.columns:
            axs_count += 1
            axs[axs_count].axhline(y=0, color='gray', linestyle='-.')
            axs[axs_count].fill_between(self.df.index, self.cmf_buffer, 
                self.df.CMF, where=self.df.CMF > self.cmf_buffer,
                interpolate=True, color='green', alpha=0.5)
            axs[axs_count].fill_between(self.df.index, -self.cmf_buffer, 
                self.df.CMF, where=self.df.CMF < -self.cmf_buffer,
                interpolate=True, color='red', alpha=0.5)
            self.df.plot(y='CMF', ax=axs[axs_count])
            axs[axs_count].legend(loc='center left')


        if 'RSI' in self.df.columns:
            axs_count += 1
            axs[axs_count].axhline(y=80, color='k', linestyle=':', alpha=0.5)
            axs[axs_count].axhline(y=self.rsi_over, color='k', linestyle=':')
            axs[axs_count].axhline(y=50, color='gray', linestyle='-.')
            axs[axs_count].axhline(y=self.rsi_under, color='k', linestyle=':')
            axs[axs_count].axhline(y=20, color='k', linestyle=':', alpha=0.5)
            axs[axs_count].fill_between(self.df.index, self.rsi_over, 
                self.df.RSI, where=self.df.RSI > self.rsi_over,
                interpolate=True, color='red', alpha=0.7)
            axs[axs_count].fill_between(self.df.index, self.rsi_under,
                self.df.RSI, where=self.df.RSI < self.rsi_under,
                interpolate=True, color='green', alpha=0.7)
            self.df.plot(y='RSI', ax=axs[axs_count])
            axs[axs_count].legend(loc='center left')


        formatter = WeekdayDateFormatter(self.df.Date)
        for ax in axs:
            # Turn on the minor TICKS, which are required for the minor GRID
            ax.minorticks_on()
            # Customize the major grid
            ax.grid(which='major', linestyle='-', linewidth='1')
            # Customize the minor grid
            ax.grid(which='minor', linestyle=':')
            ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()

        title = f'Stock {self.stock}'
        if self.stock_name is not None:
            title += f' - {self.stock_name}'
        fig.suptitle(title)
        fig.tight_layout()
        figManager = plt.get_current_fig_manager()
        figManager.window.state('zoomed') # maximize the window
        # zoom in on the last six months of data
        start_idx = 0
        if self.df.index[-1] >= 200:
            start_idx = 180
        plt.xlim(self.df.index[-start_idx], self.df.index[-1]+2)

        if save_plot:
            plt.savefig(f'img/{self.stock}.png')
        if show_plot:
            plt.show()
        plt.close()


def compare_stocks():
    """
    compare all stocks in the list.
    each stock shows percent increase and decrease by day compared to the first
    closing day of the chosen range.
    """
    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': [5, 2]})
    objs = []
    df = pd.DataFrame()
    dfv = pd.DataFrame()
    for stock in args.stock:
        obj = StockAnalysis(stock, args.start, args.end)
        day0_close = obj.df[obj.close_col].iloc[0]
        obj.df[stock] = (obj.df[obj.close_col] / day0_close - 1) * 100
        df = pd.concat([df, obj.df[[stock]]], axis=1)
        obj.df.rename(columns={'Volume': f'{stock} Vol'}, inplace=True)
        dfv = pd.concat([dfv, obj.df[[f'{stock} Vol']]], axis=1)
    # Plot close data. Rest index so numbers instead of dates can be used for x-axis
    stock_cols = df.columns.tolist()
    df.reset_index(inplace=True)
    df.plot(y=stock_cols, ax=axs[0])

    # Plot volume data. Rest index so numbers instead of dates can be used for x-axis
    vol_cols = dfv.columns.tolist()
    dfv.reset_index(inplace=True)
    for col in vol_cols:
        axs[1].plot(dfv.index, dfv[col])
        axs[1].bar(x=dfv.index, height=dfv[col], label=col, width=1, alpha=0.7)

    # format number x-axis as dates. this removes gaps in dates
    formatter = WeekdayDateFormatter(df.Date)
    for ax in axs:
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='1')
        ax.grid(which='minor', linestyle=':')
        ax.legend(loc='center left')
        ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()
    fig.suptitle(f"Comparing {args.stock}")
    fig.tight_layout()
    figManager = plt.get_current_fig_manager()
    figManager.window.state('zoomed')
    plt.show()
    plt.close()


if __name__ == "__main__":
    args = cli_parameters()
    if args.compare:
        compare_stocks()
    else:
        # stock = 'IVV'
        # start = '1/1/2020'
        # end = '5/15/2020'
        # if len(args.stock) == 0:
        #     args.stock = search_stock_symbols(None)['NASDAQ Symbol'].tolist()

        # start_stock = False
        # start_stock = True
        # args.stock = ['DOW']
        for stock in args.stock:
            obj = StockAnalysis(stock, args.start, args.end, plot_type='candlestick')
            if args.bb:
                obj.Bollinger_Bands()
            if args.macd:
                obj.MACD()
            if args.sto:
                obj.Stochastic()
            if args.rsi:
                obj.RSI()
            if args.cmf:
                obj.CMF()
            obj.plot_data(show_plot=args.show, save_plot=args.save)
            # obj.plot_data_mpf()
        if args.save:
            if len(args.stock) > 1:
                os.startfile(f'.')
            else:
                os.startfile(os.path.join('img', f'{args.stock[0]}.png'))
