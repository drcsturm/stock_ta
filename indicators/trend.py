import pandas as pd
import matplotlib

class MACD(object):
    """
    TREND INDICATOR
    Moving Average Convergence/Divergence
    close_series: pd.Series: close data with index as date
    """
    def __init__(self, close_series: pd.Series,
        slow_ema: int = 26,
        fast_ema: int = 12,
        signal_span: int = 9,
        date_series: pd.Series = None):
        self._slow_ema = slow_ema
        self._fast_ema = fast_ema
        self._signal_span = signal_span
        self._close_series = close_series
        self._date_series = date_series
        self.calc()

    def calc(self):
        self.df = pd.DataFrame()
        self.df[f'{self._slow_ema} ema'] = self._close_series.ewm(span=self._slow_ema, adjust=False).mean()
        self.df[f'{self._fast_ema} ema'] = self._close_series.ewm(span=self._fast_ema, adjust=False).mean()
        self.df['MACD'] = (self.df[f'{self._fast_ema} ema'] - self.df[f'{self._slow_ema} ema'])
        self.df['Signal'] = self.df['MACD'].ewm(span=self._signal_span, adjust=False).mean()
        self.df['Crossover'] = self.df['MACD'] - self.df['Signal']
        self.df['YCrossover'] = self.df.Crossover.shift() #yesterday crossover
        self.df['MACD_indicator'] = 0
        self.df.loc[(self.df.Crossover < 0) & (self.df.YCrossover > 0), 'MACD_indicator'] = 1 # Sell, cross line going negative
        self.df.loc[(self.df.Crossover > 0) & (self.df.YCrossover < 0), 'MACD_indicator'] = 2 # Buy, cross line going positive

    def plot(self, ax):
        # MACD buy sell indicators
        if self._date_series is not None:
            df = pd.concat([self._date_series, self._close_series], axis=1)
        else:
            df = self.df
        for index, row in df[df.MACD_indicator == 2].iterrows():
            ax.text(index, row.MACD, 'B', color='g')
        for index, row in df[df.MACD_indicator == 1].iterrows():
            ax.text(index, row.MACD, 'S', color='r')
        # MACD bars
        df["MACD Crossover diff"] = df.Crossover.diff(1)
        df["MACD bar color"] = 'r'
        df.loc[df["MACD Crossover diff"] > 0, "MACD bar color"] = 'g'
        ax.bar(df.index, df.Crossover, width=1, color=df["MACD bar color"])
        ax.axhline(y=0, color='gray', linestyle='-.')
        df.plot(y=['MACD', 'Signal', 'Crossover'], ax=ax)
        ax.legend(loc='center left')


