from flask import send_from_directory
from flask_cors import CORS
from openai import OpenAI
import statsmodels.api as sm
import webbrowser
from threading import Timer
import time
import scipy.integrate
import scipy.stats
from datetime import datetime
import yfinance as yf
import pandas as pd
from flask import Flask, request, jsonify, render_template, url_for
import matplotlib.pyplot as plt
import numpy as np
import os


client = OpenAI(api_key='Your key here')

app = Flask(__name__)
CORS(app)

scripts_dir = 'generated_scripts'
os.makedirs(scripts_dir, exist_ok=True)


@app.route('/generate-stock-selection-code', methods=['POST'])
def generate_stock_selection_code():
    data = request.json
    criteria = data.get('criteria')
    time_period = data.get('timePeriod')
    trading_strategy = data.get('tradingStrategy')

    prompt = (f"You are a coding and trading expert. Generate Python code to implement this trading strategy. "
              f"Stock selection criteria or enter stock tickers: {criteria}, "
              f"Time period: {time_period}, "
              f"Trading strategy: {trading_strategy}. "
              f"Use yfinance API to access stock data. "
              f"Please provide code to do backtest of this strategy that can be executed directly without modifications.")

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens = 2000,
        temperature = 0.5
    )

    generated_code = response.choices[0].text
    return jsonify({"generatedCode": generated_code})

@app.route('/get-script/<path:filename>')
def get_script(filename):
    return send_from_directory(scripts_dir, filename)

@app.route('/fetch-stock-data', methods=['POST'])
def fetch_stock_data():
    data = request.json
    ticker_symbol = data.get('ticker')

    # Fetch historical data for the ticker
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1mo")  # Fetching 1 month of historical data

    # Preparing data for Chart.js
    dates = [date.strftime('%Y-%m-%d') for date in hist.index]
    closing_prices = list(hist['Close'])

    return jsonify({"dates": dates, "prices": closing_prices, "data": hist.to_json()})

######################################################################################### simple moving average
def macd_method(signals, ma1, ma2):
    signals['ma1'] = signals['Close'].rolling(window=ma1, min_periods=1, center=False).mean()
    signals['ma2'] = signals['Close'].rolling(window=ma2, min_periods=1, center=False).mean()

    return signals


# signal generation
# when the short moving average is larger than long moving average, we long and hold
# when the short moving average is smaller than long moving average, we clear positions
# the logic behind this is that the momentum has more impact on short moving average
# we can subtract short moving average from long moving average
# the difference between is sometimes positive, it sometimes becomes negative
# thats why it is named as moving average converge/diverge oscillator
def signal_generation_macd(df, method, ma1, ma2):
    signals = method(df, ma1, ma2)
    signals['positions'] = 0

    # positions becomes and stays one once the short moving average is above long moving average
    signals['positions'][ma1:] = np.where(signals['ma1'][ma1:] >= signals['ma2'][ma1:], 1, 0)

    # as positions only imply the holding
    # we take the difference to generate real trade signal
    signals['signals'] = signals['positions'].diff()

    # oscillator is the difference between two moving average
    # when it is positive, we long, vice versa
    signals['oscillator'] = signals['ma1'] - signals['ma2']

    print(signals)

    return signals


# 定义保存图表的目录
images_dir = 'static/images'
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# plotting the backtesting result
def plot_macd(ma1, ma2, stdate, eddate, ticker, slicer):
    # downloading data
    df = yf.download(ticker, start=stdate, end=eddate)
    new = signal_generation_macd(df, macd_method, ma1, ma2)
    new = new[slicer:]
    # 文件名基于股票代码和时间戳来保证唯一性
    fname = f"{ticker}_{int(time.time())}.png"
    filepath = os.path.join(images_dir, fname)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plotting the first chart: Close price and LONG/SHORT positions
    new['Close'].plot(ax=ax1, label=ticker)
    ax1.plot(new.loc[new['signals'] == 1].index, new['Close'][new['signals'] == 1], label='LONG', lw=0, marker='^',
             c='g')
    ax1.plot(new.loc[new['signals'] == -1].index, new['Close'][new['signals'] == -1], label='SHORT', lw=0, marker='v',
             c='r')
    ax1.legend(loc='best')
    ax1.set_title('Positions')


    # Plotting the second chart: MACD Oscillator
    new['oscillator'].plot(ax=ax2, kind='bar', color='r')
    ax2.set_title('MACD Oscillator')
    import matplotlib.dates as mdates

    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")  # 旋转并调整位置

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    # 返回图片的相对路径或URL
    return filepath

@app.route('/macd-calculate', methods=['POST'])
def macd_calculate():
    data = request.json
    ma1 = int(data['ma1'])
    ma2 = int(data['ma2'])
    # print(ma1.type, ma2.type)
    start_date = data['start_date']
    end_date = data['end_date']
    ticker = data['ticker']
    slicer = int(data['slicer'])

    print("ok")

    image_path = plot_macd(ma1, ma2, start_date, end_date, ticker, slicer)
    image_url = url_for('static', filename=os.path.relpath(image_path, 'static'))
    return jsonify({"imageUrl": image_url})


@app.route('/macd')
def macd():
    return render_template('macd.html')


############################################################################################### Pair_trading


# use Engle-Granger two-step method to test cointegration
# the underlying method is straight forward and easy to implement
# the latest statsmodels package should ve included johansen test which is more common
# check sm.tsa.var.vecm.coint_johansen
# the malaise of two-step is the order of the cointegration
# unlike johansen test, two-step method can only detect the first order
# check the following material for further details

def EG_method(X, Y, show_summary=False):
    # step 1
    # estimate long run equilibrium
    model1 = sm.OLS(Y, sm.add_constant(X)).fit()
    epsilon = model1.resid

    if show_summary:
        print('\nStep 1\n')
        print(model1.summary())

    # check p value of augmented dickey fuller test
    # if p value is no larger than 5%, stationary test is passed
    if sm.tsa.stattools.adfuller(epsilon)[1] > 0.05:
        return False, model1

    # take first order difference of X and Y plus the lagged residual from step 1
    X_dif = sm.add_constant(pd.concat([X.diff(), epsilon.shift(1)], axis=1).dropna())
    Y_dif = Y.diff().dropna()

    # step 2
    # estimate error correction model
    model2 = sm.OLS(Y_dif, X_dif).fit()

    if show_summary:
        print('\nStep 2\n')
        print(model2.summary())

    # adjustment coefficient must be negative
    if list(model2.params)[-1] > 0:
        return False, model1
    else:
        return True, model1

# first we verify the status of cointegration by checking historical datasets
# bandwidth determines the number of data points for consideration
# bandwidth is 250 by default, around one year's data points
# if the status is valid, we check the signals
# when z stat gets above the upper bound
# we long the bearish one and short the bullish one, vice versa
def signal_generation_pairtrading(asset1, asset2, method, bandwidth=250):
    signals = pd.DataFrame()
    signals['asset1'] = asset1['Close']
    signals['asset2'] = asset2['Close']

    # signals only imply holding
    signals['signals1'] = 0
    signals['signals2'] = 0

    # initialize
    prev_status = False
    signals['z'] = np.nan
    signals['z upper limit'] = np.nan
    signals['z lower limit'] = np.nan
    signals['fitted'] = np.nan
    signals['residual'] = np.nan

    # signal processing
    for i in range(bandwidth, len(signals)):

        # cointegration test
        coint_status, model = method(signals['asset1'].iloc[i - bandwidth:i],
                                     signals['asset2'].iloc[i - bandwidth:i])

        # cointegration breaks
        # clear existing positions
        if prev_status and not coint_status:
            if signals.at[signals.index[i - 1], 'signals1'] != 0:
                signals.at[signals.index[i], 'signals1'] = 0
                signals.at[signals.index[i], 'signals2'] = 0
                signals['z'].iloc[i:] = np.nan
                signals['z upper limit'].iloc[i:] = np.nan
                signals['z lower limit'].iloc[i:] = np.nan
                signals['fitted'].iloc[i:] = np.nan
                signals['residual'].iloc[i:] = np.nan

        # cointegration starts
        # set the trigger conditions
        # this is no forward bias
        # just to minimize the calculation done in pandas
        if not prev_status and coint_status:
            # predict the price to compute the residual
            signals['fitted'].iloc[i:] = model.predict(sm.add_constant(signals['asset1'].iloc[i:]))
            signals['residual'].iloc[i:] = signals['asset2'].iloc[i:] - signals['fitted'].iloc[i:]

            # normalize the residual to get z stat
            # z should be a white noise following N(0,1)
            signals['z'].iloc[i:] = (signals['residual'].iloc[i:] - np.mean(model.resid)) / np.std(model.resid)

            # create thresholds
            # conventionally one sigma is the threshold
            # two sigma reaches 95% which is relatively difficult to trigger
            signals['z upper limit'].iloc[i:] = signals['z'].iloc[i] + np.std(model.resid)
            signals['z lower limit'].iloc[i:] = signals['z'].iloc[i] - np.std(model.resid)

        # as z stat cannot exceed both upper and lower bounds at the same time
        # the lines below hold
        if coint_status and signals['z'].iloc[i] > signals['z upper limit'].iloc[i]:
            signals.at[signals.index[i], 'signals1'] = 1
        if coint_status and signals['z'].iloc[i] < signals['z lower limit'].iloc[i]:
            signals.at[signals.index[i], 'signals1'] = -1

        prev_status = coint_status

        # signals only imply holding
    # we take the first order difference to obtain the execution signal
    signals['positions1'] = signals['signals1'].diff()

    # only need to generate trading signal of one asset
    # the other one should be the opposite direction
    signals['signals2'] = -signals['signals1']
    signals['positions2'] = signals['signals2'].diff()

    return signals


def plot_pairtrading(data, ticker1, ticker2):
    fig = plt.figure(figsize=(10, 5))
    bx = fig.add_subplot(111)
    bx2 = bx.twinx()

    # viz two different assets
    asset1_price, = bx.plot(data.index, data['asset1'],
                            c='#113aac', alpha=0.7)
    asset2_price, = bx2.plot(data.index, data['asset2'],
                             c='#907163', alpha=0.7)

    # viz positions
    asset1_long, = bx.plot(data.loc[data['positions1'] == 1].index,
                           data['asset1'][data['positions1'] == 1],
                           lw=0, marker='^', markersize=8,
                           c='g', alpha=0.7)
    asset1_short, = bx.plot(data.loc[data['positions1'] == -1].index,
                            data['asset1'][data['positions1'] == -1],
                            lw=0, marker='v', markersize=8,
                            c='r', alpha=0.7)
    asset2_long, = bx2.plot(data.loc[data['positions2'] == 1].index,
                            data['asset2'][data['positions2'] == 1],
                            lw=0, marker='^', markersize=8,
                            c='g', alpha=0.7)
    asset2_short, = bx2.plot(data.loc[data['positions2'] == -1].index,
                             data['asset2'][data['positions2'] == -1],
                             lw=0, marker='v', markersize=8,
                             c='r', alpha=0.7)

    # set labels
    bx.set_ylabel(ticker1, )
    bx2.set_ylabel(ticker2, rotation=270)
    bx.yaxis.labelpad = 15
    bx2.yaxis.labelpad = 15
    bx.set_xlabel('Date')
    bx.xaxis.labelpad = 15

    plt.legend([asset1_price, asset2_price, asset1_long, asset1_short],
               [ticker1, ticker2,
                'LONG', 'SHORT'],
               loc='lower left')

    plt.title('Pair Trading')
    plt.xlabel('Date')
    plt.grid(True)
    fname = f"{ticker1}_{ticker2}_{int(time.time())}.png"
    filepath = os.path.join(images_dir, fname)
    plt.savefig(filepath)
    plt.close()

    # 返回图片的相对路径或URL
    return filepath

# visualize overall portfolio performance
def portfolio_pairtrading(data):
    # initial capital to calculate the actual pnl
    capital0 = 20000

    # shares to buy of each position
    # this is no forward bias
    # just ensure we have enough €€€ to purchase shares when the price peaks
    positions1 = capital0 // max(data['asset1'])
    positions2 = capital0 // max(data['asset2'])

    # cumsum1 column is created to check the holding of the position
    data['cumsum1'] = data['positions1'].cumsum()

    # since there are two assets, we calculate each asset separately
    # in the end we aggregate them into one portfolio
    portfolio = pd.DataFrame()
    portfolio['asset1'] = data['asset1']
    portfolio['holdings1'] = data['cumsum1'] * data['asset1'] * positions1
    portfolio['cash1'] = capital0 - (data['positions1'] * data['asset1'] * positions1).cumsum()
    portfolio['total asset1'] = portfolio['holdings1'] + portfolio['cash1']
    portfolio['return1'] = portfolio['total asset1'].pct_change()
    portfolio['positions1'] = data['positions1']

    data['cumsum2'] = data['positions2'].cumsum()
    portfolio['asset2'] = data['asset2']
    portfolio['holdings2'] = data['cumsum2'] * data['asset2'] * positions2
    portfolio['cash2'] = capital0 - (data['positions2'] * data['asset2'] * positions2).cumsum()
    portfolio['total asset2'] = portfolio['holdings2'] + portfolio['cash2']
    portfolio['return2'] = portfolio['total asset2'].pct_change()
    portfolio['positions2'] = data['positions2']

    portfolio['z'] = data['z']
    portfolio['total asset'] = portfolio['total asset1'] + portfolio['total asset2']
    portfolio['z upper limit'] = data['z upper limit']
    portfolio['z lower limit'] = data['z lower limit']

    # plotting the asset value change of the portfolio
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    total_asset_performance, = ax.plot(portfolio['total asset'], c='#46344e')
    z_stats, = ax2.plot(portfolio['z'], c='#4f4a41', alpha=0.2)

    threshold = ax2.fill_between(portfolio.index, portfolio['z upper limit'],
                                 portfolio['z lower limit'],
                                 alpha=0.2, color='#ffb48f')

    # due to the opposite direction of trade for 2 assets
    # we will not plot positions on asset performance
    ax.set_ylabel('Asset Value')
    ax2.set_ylabel('Z Statistics', rotation=270)
    ax.yaxis.labelpad = 15
    ax2.yaxis.labelpad = 15
    ax.set_xlabel('Date')
    ax.xaxis.labelpad = 15

    plt.legend([z_stats, threshold, total_asset_performance],
               ['Z Statistics', 'Z Statistics +-1 Sigma',
                'Total Asset Performance'], loc='best')

    plt.grid(True)
    plt.title('Total Asset')

    fname = f"Portfolio_{int(time.time())}.png"
    filepath = os.path.join(images_dir, fname)
    plt.savefig(filepath)
    plt.close()

    return filepath

@app.route('/pair_trading_calculate', methods=['POST'])
def pair_trading_calculate():
    data = request.json
    # the sample i am using are NVDA and AMD from 2013 to 2014
    stdate = data['start_date']
    eddate = data['end_date']
    ticker1 = data['ticker1']
    ticker2 = data['ticker2']
    # extract data
    asset1 = yf.download(ticker1, start=stdate, end=eddate)
    asset2 = yf.download(ticker2, start=stdate, end=eddate)

    # create signals
    try:
        signals = signal_generation_pairtrading(asset1, asset2, EG_method)
        ind = signals['z'].dropna().index[0]
        # Visualization and further processing here...
        messages = "Data processed successfully"
        image_path = plot_pairtrading(signals[ind:], ticker1, ticker2)
        portfolio_path = portfolio_pairtrading(signals[ind:])
        image_url = url_for('static', filename=os.path.relpath(image_path, 'static'))
        images_url = url_for('static', filename=os.path.relpath(portfolio_path, 'static'))

        # return render_template('pair_trading.html', message=message), jsonify({"imageUrl": image_url})
        return jsonify({"imageUrl": image_url, "portfolio_url": images_url, "messages": messages})
    except IndexError:
        messages = "Error: No trading Signal"
        return jsonify({"messages": messages})
    except Exception as e:
        messages = f"An error occurred: {str(e)}"
        return jsonify({"messages": messages})



@app.route('/Pair_trading')
def Pair_trading():
    return render_template('Pair_trading.html')



############################################################################################## Heikin_Ashi


# Heikin Ashi has a unique method to filter out the noise
# its open, close, high, low require a different approach
# please refer to the website mentioned above
def heikin_ashi(data):
    df = data.copy()

    df.reset_index(inplace=True)

    # heikin ashi close
    df['HA close'] = (df['Open'] + df['Close'] + df['High'] + df['Low']) / 4

    # initialize heikin ashi open
    df['HA open'] = float(0)
    df['HA open'][0] = df['Open'][0]

    # heikin ashi open
    for n in range(1, len(df)):
        df.at[n, 'HA open'] = (df['HA open'][n - 1] + df['HA close'][n - 1]) / 2

    # heikin ashi high/low
    temp = pd.concat([df['HA open'], df['HA close'], df['Low'], df['High']], axis=1)
    df['HA high'] = temp.apply(max, axis=1)
    df['HA low'] = temp.apply(min, axis=1)

    del df['Adj Close']
    del df['Volume']

    return df


# setting up signal generations
# trigger conditions can be found from the website mentioned above
# they kinda look like marubozu candles
# there s a short strategy as well
# the trigger condition of short strategy is the reverse of long strategy
# you have to satisfy all four conditions to long/short
# nevertheless, the exit signal only has three conditions
def signal_generation_HA(df, method, stls):
    data = method(df)

    data['signals'] = 0

    # i use cumulated sum to check how many positions i have longed
    # i would ignore the exit signal prior if not holding positions
    # i also keep tracking how many long positions i have got
    # long signals cannot exceed the stop loss limit
    data['cumsum'] = 0

    for n in range(1, len(data)):

        # long triggered
        if (data['HA open'][n] > data['HA close'][n] and data['HA open'][n] == data['HA high'][n] and
                np.abs(data['HA open'][n] - data['HA close'][n]) > np.abs(
                    data['HA open'][n - 1] - data['HA close'][n - 1]) and
                data['HA open'][n - 1] > data['HA close'][n - 1]):

            data.at[n, 'signals'] = 1
            data['cumsum'] = data['signals'].cumsum()

            # accumulate too many longs
            if data['cumsum'][n] > stls:
                data.at[n, 'signals'] = 0

        # exit positions
        elif (data['HA open'][n] < data['HA close'][n] and data['HA open'][n] == data['HA low'][n] and
              data['HA open'][n - 1] < data['HA close'][n - 1]):

            data.at[n, 'signals'] = -1
            data['cumsum'] = data['signals'].cumsum()

            # clear all longs
            # if there are no long positions in my portfolio
            # ignore the exit signal
            if data['cumsum'][n] > 0:
                data.at[n, 'signals'] = -1 * (data['cumsum'][n - 1])

            if data['cumsum'][n] < 0:
                data.at[n, 'signals'] = 0

    return data

# since matplotlib remove the candlestick
# plus we dont wanna install mpl_finance
# we implement our own version
# simply use fill_between to construct the bar
# use line plot to construct high and low
def candlestick(df, ax=None, titlename='', highcol='High', lowcol='Low',
                opencol='Open', closecol='Close', xcol='Date',
                colorup='r', colordown='g', **kwargs):
    # bar width
    # use 0.6 by default
    dif = [(-3 + i) / 10 for i in range(7)]

    if not ax:
        ax = plt.figure(figsize=(10, 5)).add_subplot(111)

    # construct the bars one by one
    for i in range(len(df)):

        # width is 0.6 by default
        # so 7 data points required for each bar
        x = [i + j for j in dif]
        y1 = [df[opencol].iloc[i]] * 7
        y2 = [df[closecol].iloc[i]] * 7

        barcolor = colorup if y1[0] > y2[0] else colordown

        # no high line plot if open/close is high
        if df[highcol].iloc[i] != max(df[opencol].iloc[i], df[closecol].iloc[i]):
            # use generic plot to viz high and low
            # use 1.001 as a scaling factor
            # to prevent high line from crossing into the bar
            plt.plot([i, i],
                     [df[highcol].iloc[i],
                      max(df[opencol].iloc[i],
                          df[closecol].iloc[i]) * 1.001], c='k', **kwargs)

        # same as high
        if df[lowcol].iloc[i] != min(df[opencol].iloc[i], df[closecol].iloc[i]):
            plt.plot([i, i],
                     [df[lowcol].iloc[i],
                      min(df[opencol].iloc[i],
                          df[closecol].iloc[i]) * 0.999], c='k', **kwargs)

        # treat the bar as fill between
        plt.fill_between(x, y1, y2,
                         edgecolor='k',
                         facecolor=barcolor, **kwargs)

    # only show 5 xticks
    step_size = max(1, len(df) // 5)
    plt.xticks(range(0, len(df), step_size), df[xcol][0::step_size].dt.date)
    plt.title(titlename)


# plotting the backtesting result
def plot_HA(df, ticker):
    df.set_index(df['Date'], inplace=True)

    # first plot is Heikin-Ashi candlestick
    # use candlestick function and set Heikin-Ashi O,C,H,L
    ax1 = plt.subplot2grid((200, 1), (0, 0), rowspan=120, ylabel='HA price')
    candlestick(df, ax1, titlename='', highcol='HA high', lowcol='HA low',
                opencol='HA open', closecol='HA close', xcol='Date',
                colorup='r', colordown='g')
    plt.grid(True)
    plt.xticks([])
    plt.title('Heikin-Ashi')
    fname = f"{ticker}_{int(time.time())}_'HA'.png"
    filepath = os.path.join(images_dir, fname)
    # the second plot is the actual price with long/short positions as up/down arrows
    ax2 = plt.subplot2grid((200, 1), (120, 0), rowspan=80, ylabel='price', xlabel='')
    df['Close'].plot(ax=ax2, label=ticker)

    # long/short positions are attached to the real close price of the stock
    # set the line width to zero
    # thats why we only observe markers
    ax2.plot(df.loc[df['signals'] == 1].index, df['Close'][df['signals'] == 1], marker='^', lw=0, c='g', label='long')
    ax2.plot(df.loc[df['signals'] < 0].index, df['Close'][df['signals'] < 0], marker='v', lw=0, c='r', label='short')

    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(filepath)
    plt.close()

    # 返回图片的相对路径或URL
    return filepath

# backtesting
# initial capital 10k to calculate the actual pnl
# 100 shares to buy of every position
def portfolio(data, capital0=10000, positions=100):
    # cumsum column is created to check the holding of the position
    data['cumsum'] = data['signals'].cumsum()

    portfolio = pd.DataFrame()
    portfolio['holdings'] = data['cumsum'] * data['Close'] * positions
    portfolio['cash'] = capital0 - (data['signals'] * data['Close'] * positions).cumsum()
    portfolio['total asset'] = portfolio['holdings'] + portfolio['cash']
    portfolio['return'] = portfolio['total asset'].pct_change()
    portfolio['signals'] = data['signals']
    portfolio['date'] = data['Date']
    portfolio.set_index('date', inplace=True)

    return portfolio



# plotting the asset value change of the portfolio
def profit(portfolio):
    fig = plt.figure()
    bx = fig.add_subplot(111)

    portfolio['total asset'].plot(label='Total Asset')

    # long/short position markers related to the portfolio
    # the same mechanism as the previous one
    # replace close price with total asset value
    bx.plot(portfolio['signals'].loc[portfolio['signals'] == 1].index,
            portfolio['total asset'][portfolio['signals'] == 1], lw=0, marker='^', c='g', label='long')
    bx.plot(portfolio['signals'].loc[portfolio['signals'] < 0].index,
            portfolio['total asset'][portfolio['signals'] < 0], lw=0, marker='v', c='r', label='short')

    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Asset Value')
    plt.title('Total Asset')

    fname = f"Portfolio_{int(time.time())}.png"
    filepath = os.path.join(images_dir, fname)
    plt.savefig(filepath)
    plt.close()

    # 返回图片的相对路径或URL
    return filepath

def omega(risk_free, degree_of_freedom, maximum, minimum):
    y = scipy.integrate.quad(lambda g: 1 - scipy.stats.t.cdf(g, degree_of_freedom), risk_free, maximum)
    x = scipy.integrate.quad(lambda g: scipy.stats.t.cdf(g, degree_of_freedom), minimum, risk_free)

    z = (y[0]) / (x[0])

    return z


# sortino ratio is another variation of sharpe ratio
# the standard deviation of all returns is substituted with standard deviation of negative returns
# sortino ratio measures the impact of negative return on return
# i am also using student T probability distribution function instead of normal distribution
# check wikipedia for more details
# https://en.wikipedia.org/wiki/Sortino_ratio
def sortino(risk_free, degree_of_freedom, growth_rate, minimum):
    v = np.sqrt(np.abs(
        scipy.integrate.quad(lambda g: ((risk_free - g) ** 2) * scipy.stats.t.pdf(g, degree_of_freedom), risk_free,
                             minimum)))
    s = (growth_rate - risk_free) / v[0]

    return s


# i use a function to calculate maximum drawdown
# the idea is simple
# for every day, we take the current asset value marked to market
# to compare with the previous highest asset value
# we get our daily drawdown
# it is supposed to be negative if the current one is not the highest
# we implement a temporary variable to store the minimum negative value
# which is called maximum drawdown
# for each daily drawdown that is smaller than our temporary value
# we update the temp until we finish our traversal
# in the end we return the maximum drawdown
def mdd(series):
    minimum = 0
    for i in range(1, len(series)):
        if minimum > (series[i] / max(series[:i]) - 1):
            minimum = (series[i] / max(series[:i]) - 1)

    return minimum


# stats calculation
def stats(portfolio, trading_signals, stdate, eddate, capital0=10000):
    stats = pd.DataFrame([0])

    # get the min and max of return
    maximum = np.max(portfolio['return'])
    minimum = np.min(portfolio['return'])

    # growth_rate denotes the average growth rate of portfolio
    # use geometric average instead of arithmetic average for percentage growth
    if portfolio.empty:
        print("Error: Portfolio data is empty.")
        return  # Or handle it in another appropriate way
    if len(trading_signals) == 0:
        print("Error: No trading signals available.")
        return  # Or handle it differently
    last_total_asset = portfolio['total asset'].iloc[-1] if not portfolio.empty else 0
    growth_rate = (float(last_total_asset / capital0)) ** (1 / max(1, len(trading_signals))) - 1

    # calculating the standard deviation
    std = float(np.sqrt((((portfolio['return'] - growth_rate) ** 2).sum()) / len(trading_signals)))

    # use S&P500 as benchmark
    benchmark = yf.download('^GSPC', start=stdate, end=eddate)

    # return of benchmark
    return_of_benchmark = float(benchmark['Close'].iloc[-1] / benchmark['Open'].iloc[0] - 1)

    # rate_of_benchmark denotes the average growth rate of benchmark
    # use geometric average instead of arithmetic average for percentage growth
    rate_of_benchmark = (return_of_benchmark + 1) ** (1 / len(trading_signals)) - 1

    del benchmark

    # backtesting stats
    # CAGR stands for cumulated average growth rate
    stats['CAGR'] = stats['portfolio return'] = float(0)
    stats['CAGR'][0] = growth_rate
    stats['portfolio return'][0] = portfolio['total asset'].iloc[-1] / capital0 - 1
    stats['benchmark return'] = return_of_benchmark
    stats['sharpe ratio'] = (growth_rate - rate_of_benchmark) / std
    stats['maximum drawdown'] = mdd(portfolio['total asset'])

    # calmar ratio is sorta like sharpe ratio
    # the standard deviation is replaced by maximum drawdown
    # it is the measurement of return after worse scenario adjustment
    # check wikipedia for more details
    # https://en.wikipedia.org/wiki/Calmar_ratio
    stats['calmar ratio'] = growth_rate / stats['maximum drawdown']
    stats['omega ratio'] = omega(rate_of_benchmark, len(trading_signals), maximum, minimum)
    stats['sortino ratio'] = sortino(rate_of_benchmark, len(trading_signals), growth_rate, minimum)

    # note that i use stop loss limit to limit the numbers of longs
    # and when clearing positions, we clear all the positions at once
    # so every long is always one, and short cannot be larger than the stop loss limit
    stats['numbers of longs'] = trading_signals['signals'].loc[trading_signals['signals'] == 1].count()
    stats['numbers of shorts'] = trading_signals['signals'].loc[trading_signals['signals'] < 0].count()
    stats['numbers of trades'] = stats['numbers of shorts'] + stats['numbers of longs']

    # to get the total length of trades
    # given that cumsum indicates the holding of positions
    # we can get all the possible outcomes when cumsum doesnt equal zero
    # then we count how many non-zero positions there are
    # we get the estimation of total length of trades
    stats['total length of trades'] = trading_signals['signals'].loc[trading_signals['cumsum'] != 0].count()
    stats['average length of trades'] = stats['total length of trades'] / stats['numbers of trades']
    stats['profit per trade'] = float(0)
    stats['profit per trade'].iloc[0] = (portfolio['total asset'].iloc[-1] - capital0) / \
                                        stats['numbers of trades'].iloc[0]

    del stats[0]
    print(stats)

@app.route('/Heikin_Ashi_calculate', methods=['POST'])
def Heikin_Ashi_calculate():
    # initializing
    data = request.json
    # stop loss positions, the maximum long positions we can get
    # without certain constraints, you will long indefinites times
    # as long as the market condition triggers the signal
    # in a whipsaw condition, it is suicidal
    stls =int( data['stls'])
    ticker = data['ticker']
    stdate_str = data['start_date']
    eddate_str = data['end_date']
    stdate = datetime.strptime(stdate_str, '%Y-%m-%d')
    eddate = datetime.strptime(eddate_str, '%Y-%m-%d')

    # slicer is used for plotting
    # a three year dataset with 750 data points would be too much
    slicer = 700

    # downloading data
    df = yf.download(ticker, start=stdate, end=eddate)
    trading_signals = signal_generation_HA(df, heikin_ashi, stls)

    if trading_signals['signals'].sum() == 0:
        print("No trading signals generated. Check the signal conditions.")

    viz = trading_signals[slicer:]

    portfolio_details = portfolio(viz)
    stats(portfolio_details, trading_signals, stdate, eddate)

    porfit_path = profit(portfolio_details)
    image_path = plot_HA(viz, ticker)

    image_url = url_for('static', filename=os.path.relpath(image_path, 'static'))
    images_url = url_for('static', filename=os.path.relpath(porfit_path, 'static'))

    return jsonify({"imageUrl": image_url, "portfit_url": images_url})
    # note that this is the only py file with complete stats calculation


@app.route('/Heikin_Ashi')
def Heikin_Ashi():
    return render_template('Heikin_Ashi.html')

##################################################################################################### RSI

def smma(series, n):
    output = [series[0]]

    for i in range(1, len(series)):
        temp = output[-1] * (n - 1) + series[i]
        output.append(temp / n)

    return output


# calculating rsi is very simple
# except there are several versions of moving average for rsi
# simple moving average, exponentially weighted moving average, etc
# in this script, we use smoothed moving average(the authentic way)
def rsi(data, n=14):
    delta = data.diff().dropna()

    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)

    rs = np.divide(smma(up, n), smma(down, n))

    output = 100 - 100 / (1 + rs)

    return output[n - 1:]


# signal generation
# it is really easy
# when rsi goes above 70, we short the stock
# we bet the stock price would fall
# vice versa
def signal_generation_RSI(df, method, n=14):
    df['rsi'] = 0.0
    df['rsi'][n:] = method(df['Close'], n=14)

    df['positions'] = np.select([df['rsi'] < 30, df['rsi'] > 70], \
                                [1, -1], default=0)
    df['signals'] = df['positions'].diff()

    return df[n:]


# plotting
def plot_RSI(new, ticker):
    # the first plot is the actual close price with long/short positions
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(211)
    new['Close'].plot(label=ticker)
    ax.plot(new.loc[new['signals'] == 1].index,
            new['Close'][new['signals'] == 1],
            label='LONG', lw=0, marker='^', c='g')
    ax.plot(new.loc[new['signals'] == -1].index,
            new['Close'][new['signals'] == -1],
            label='SHORT', lw=0, marker='v', c='r')

    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Positions')
    plt.xlabel('Date')
    plt.ylabel('price')
    # plt.savefig(filepath1)


    # the second plot is rsi with overbought/oversold interval capped at 30/70
    bx = fig.add_subplot(212, sharex=ax)
    new['rsi'].plot(label='relative strength index', c='#522e75')
    bx.fill_between(new.index, 30, 70, alpha=0.5, color='#f22f08')

    bx.text(new.index[-45], 75, 'overbought', color='#594346', size=12.5)
    bx.text(new.index[-45], 25, 'oversold', color='#594346', size=12.5)

    plt.xlabel('Date')
    plt.ylabel('value')
    plt.title('RSI')
    plt.legend(loc='best')
    plt.grid(True)

    fname = f"{ticker}_{int(time.time())}_'RSI'.png"
    filepath = os.path.join(images_dir, fname)
    plt.savefig(filepath)
    plt.close(fig)
    print("plot_RSI")
    # 返回图片的相对路径或URL
    return filepath


# pattern recognition
# do u really think i would write such an easy script?
# dont be naive, here is another way of using rsi
# unlike double bottom pattern for bollinger bands
# this is head-shoulder pattern directly on rsi instead of price
# well, it is actually named head and shoulders
# but i refused to do free marketing for the shampoo
# cuz that shampoo doesnt work at all!
# the details of head-shoulder pattern could be found in this link
# https://www.investopedia.com/terms/h/head-shoulders.asp

# any way, this pattern recognition is similar to the one in bollinger bands
# plz refer to bollinger bands for a detailed explanation
# https://github.com/je-suis-tm/quant-trading/blob/master/Bollinger%20Bands%20Pattern%20Recognition%20backtest.py
def pattern_recognition(df, method, lag=14):
    df['rsi'] = 0.0
    df['rsi'][lag:] = method(df['Close'], lag)

    # as usual, period is defined as the horizon for finding the pattern
    period = 25

    # delta is the threshold of the difference between two prices
    # if the difference is smaller than delta
    # we can conclude two prices are not significantly different from each other
    # the significant level is defined as delta
    delta = 0.2

    # these are the multipliers of delta
    # we wanna make sure there is head and shoulders are significantly larger than other nodes
    # the significant level is defined as head/shoulder multiplier*delta
    head = 1.1
    shoulder = 1.1

    df['signals'] = 0
    df['cumsum'] = 0
    df['coordinates'] = ''

    # now these are the parameters set by us based on experience
    # entry_rsi is the rsi when we enter a trade
    # we would exit the trade based on two conditions
    # one is that we hold the stock for more than five days
    # the variable for five days is called exit_days
    # we use a variable called counter to keep track of it
    # two is that rsi has increased more than 4 since the entry
    # the variable for 4 is called exit_rsi
    # when either condition is triggered, we exit the trade
    # this is a lazy way to exit the trade
    # cuz i dont wanna import indicators from other scripts
    # i would suggest people to use other indicators such as macd or bollinger bands
    # exiting trades based on rsi is definitely inefficient and unprofitable
    entry_rsi = 0.0
    counter = 0
    exit_rsi = 4
    exit_days = 5

    # signal generation
    # plz refer to the following link for pattern visualization
    # the idea is to start with the first node i
    # we look backwards and find the head node j with maximum value in pattern finding period
    # between node i and node j, we find a node k with its value almost the same as node i
    # started from node j to left, we find a node l with its value almost the same as node i
    # between the left beginning and node l, we find a node m with its value almost the same as node i
    # after that, we find the shoulder node n with maximum value between node m and node l
    # finally, we find the shoulder node o with its value almost the same as node n
    for i in range(period + lag, len(df)):

        # this is pretty much the same idea as in bollinger bands
        # except we have two variables
        # one for shoulder and one for the bottom nodes
        moveon = False
        top = 0.0
        bottom = 0.0

        # we have to make sure no holding positions
        # and the close price is not the maximum point of pattern finding horizon
        if (df['cumsum'][i] == 0) and \
                (df['Close'][i] != max(df['Close'][i - period:i])):

            # get the head node j with maximum value in pattern finding period
            # note that dataframe is in datetime index
            # we wanna convert the result of idxmax to a numerical index number
            j = df.index.get_loc(df['Close'][i - period:i].idxmax())

            # if the head node j is significantly larger than node i
            # we would move on to the next phrase
            if (np.abs(df['Close'][j] - df['Close'][i]) > head * delta):
                bottom = df['Close'][i]
                moveon = True

            # we try to find node k between node j and node i
            # if node k is not significantly different from node i
            # we would move on to the next phrase
            if moveon == True:
                moveon = False
                for k in range(j, i):
                    if (np.abs(df['Close'][k] - bottom) < delta):
                        moveon = True
                        break

            # we try to find node l between node j and the end of pattern finding horizon
            # note that we start from node j to the left
            # cuz we need to find another bottom node m later which would start from the left beginning
            # this way we can make sure we would find a shoulder node n between node m and node l
            # if node l is not significantly different from node i
            # we would move on to the next phrase
            if moveon == True:
                moveon = False
                for l in range(j, i - period + 1, -1):
                    if (np.abs(df['Close'][l] - bottom) < delta):
                        moveon = True
                        break

            # we try to find node m between node l and the end of pattern finding horizon
            # this time we start from left to right as usual
            # if node m is not significantly different from node i
            # we would move on to the next phrase
            if moveon == True:
                moveon = False
                for m in range(i - period, l):
                    if (np.abs(df['Close'][m] - bottom) < delta):
                        moveon = True
                        break

            # get the shoulder node n with maximum value between node m and node l
            # note that dataframe is in datetime index
            # we wanna convert the result of idxmax to a numerical index number
            # if node n is significantly larger than node i and significantly smaller than node j
            # we would move on to the next phrase
            if moveon == True:
                moveon = False
                n = df.index.get_loc(df['Close'][m:l].idxmax())
                if (df['Close'][n] - bottom > shoulder * delta) and \
                        (df['Close'][j] - df['Close'][n] > shoulder * delta):
                    top = df['Close'][n]
                    moveon = True

            # we try to find shoulder node o between node k and node i
            # if node o is not significantly different from node n
            # we would set up the signals and coordinates for visualization
            # we also need to refresh cumsum and entry_rsi for exiting the trade
            # note that moveon is still set as True
            # it would help the algo to ignore this round of iteration for exiting the trade
            if moveon == True:
                for o in range(k, i):
                    if (np.abs(df['Close'][o] - top) < delta):
                        df.at[df.index[i], 'signals'] = -1
                        df.at[df.index[i], 'coordinates'] = '%s,%s,%s,%s,%s,%s,%s' % (m, n, l, j, k, o, i)
                        df['cumsum'] = df['signals'].cumsum()
                        entry_rsi = df['rsi'][i]
                        moveon = True
                        break

        # each time we have a holding position
        # counter would steadily increase
        # if either of the exit conditions is met
        # we exit the trade with long position
        # and we refresh counter, entry_rsi and cumsum
        # you may wonder why do we need cumsum?
        # well, this is for holding positions in case you wanna check on portfolio performance
        if entry_rsi != 0 and moveon == False:
            counter += 1
            if (df['rsi'][i] - entry_rsi > exit_rsi) or \
                    (counter > exit_days):
                df.at[df.index[i], 'signals'] = 1
                df['cumsum'] = df['signals'].cumsum()
                counter = 0
                entry_rsi = 0

    return df


# visualize the pattern

@app.route('/RSI_Calculate', methods=['POST'])
def RSI_Calculate():
    data = request.json
    ticker = data['ticker']
    startdate = data['start_date']
    enddate = data['end_date']
    df = yf.download(ticker, start=startdate, end=enddate)
    print('df yes')
    new = signal_generation_RSI(df, rsi, n=14)
    print('new yes')
    image_path = plot_RSI(new, ticker)
    image_url = url_for('static', filename=os.path.relpath(image_path, 'static'))
    return jsonify({"imageUrl": image_url})

@app.route('/RSI')
def RSI():
    return render_template('RSI.html')



############################################################################################## Parabolic

def parabolic_sar(new):
    # this is common accelerating factors for forex and commodity
    # for equity, af for each step could be set to 0.01
    initial_af = 0.02
    step_af = 0.02
    end_af = 0.2

    new['trend'] = 0
    new['sar'] = 0.0
    new['real sar'] = 0.0
    new['ep'] = 0.0
    new['af'] = 0.0

    # initial values for recursive calculation
    new['trend'][1] = 1 if new['Close'][1] > new['Close'][0] else -1
    new['sar'][1] = new['High'][0] if new['trend'][1] > 0 else new['Low'][0]
    new.at[1, 'real sar'] = new['sar'][1]
    new['ep'][1] = new['High'][1] if new['trend'][1] > 0 else new['Low'][1]
    new['af'][1] = initial_af

    # calculation
    for i in range(2, len(new)):

        temp = new['sar'][i - 1] + new['af'][i - 1] * (new['ep'][i - 1] - new['sar'][i - 1])
        if new['trend'][i - 1] < 0:
            new.at[i, 'sar'] = max(temp, new['High'][i - 1], new['High'][i - 2])
            temp = 1 if new['sar'][i] < new['High'][i] else new['trend'][i - 1] - 1
        else:
            new.at[i, 'sar'] = min(temp, new['Low'][i - 1], new['Low'][i - 2])
            temp = -1 if new['sar'][i] > new['Low'][i] else new['trend'][i - 1] + 1
        new.at[i, 'trend'] = temp

        if new['trend'][i] < 0:
            temp = min(new['Low'][i], new['ep'][i - 1]) if new['trend'][i] != -1 else new['Low'][i]
        else:
            temp = max(new['High'][i], new['ep'][i - 1]) if new['trend'][i] != 1 else new['High'][i]
        new.at[i, 'ep'] = temp

        if np.abs(new['trend'][i]) == 1:
            temp = new['ep'][i - 1]
            new.at[i, 'af'] = initial_af
        else:
            temp = new['sar'][i]
            if new['ep'][i] == new['ep'][i - 1]:
                new.at[i, 'af'] = new['af'][i - 1]
            else:
                new.at[i, 'af'] = min(end_af, new['af'][i - 1] + step_af)
        new.at[i, 'real sar'] = temp

    return new


# generating signals
# idea is the same as macd oscillator
# check the website below to learn more
# https://github.com/je-suis-tm/quant-trading/blob/master/MACD%20oscillator%20backtest.py

def signal_generation_parabolic(df, method):
    new = method(df)

    new['positions'], new['signals'] = 0, 0
    new['positions'] = np.where(new['real sar'] < new['Close'], 1, 0)
    new['signals'] = new['positions'].diff()

    return new


# plotting of sar and trading positions
# still similar to macd

def plot_parabolic(new, ticker):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    new['Close'].plot(lw=3, label='%s' % ticker)
    new['real sar'].plot(linestyle=':', label='Parabolic SAR', color='k')
    ax.plot(new.loc[new['signals'] == 1].index, new['Close'][new['signals'] == 1], marker='^', color='g', label='LONG',
            lw=0, markersize=10)
    ax.plot(new.loc[new['signals'] == -1].index, new['Close'][new['signals'] == -1], marker='v', color='r',
            label='SHORT', lw=0, markersize=10)

    plt.legend()
    plt.grid(True)
    plt.title('Parabolic SAR')
    plt.ylabel('price')
    fname = f"{ticker}_{int(time.time())}_'parabolic'.png"
    filepath = os.path.join(images_dir, fname)
    plt.savefig(filepath)
    plt.close()

    return filepath

@app.route('/Parabolic_Calculate', methods=['POST'])
def Parabolic_Calculate():
    # download data via fix yahoo finance library
    data = request.get_json()
    stdate = data['start_date']
    eddate = data['end_date']
    ticker = data['ticker']

    # slice is used for plotting
    # a two year dataset with 500 variables would be too much for a figure
    slicer = 450

    df = yf.download(ticker, start=stdate, end=eddate)

    # delete adj close and volume
    # as we don't need them
    del df['Adj Close']
    del df['Volume']

    # no need to iterate over timestamp index
    df.reset_index(inplace=True)

    new = signal_generation_parabolic(df, parabolic_sar)
    print(new.head())
    # convert back to time series for plotting
    # so that we get a date x axis
    new.set_index(new['Date'], inplace=True)

    # shorten our plotting horizon and plot
    new = new[slicer:]

    image_path = plot_parabolic(new, ticker)

    image_url = url_for('static', filename=os.path.relpath(image_path, 'static'))

    # return render_template('pair_trading.html', message=message), jsonify({"imageUrl": image_url})
    return jsonify({"imageUrl": image_url})


@app.route('/Parabolic')
def Pair_tradings():
    return render_template('Parabolic.html')

@app.route('/')
def index():
    return render_template('index.html')  # Serve index.html file

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True)
