
# parabolic stop and reverse is very useful for trend following
# sar is an indicator below the price when its an uptrend
# and above the price when its a downtrend
# it is very painful to calculate sar, though
# and many explanations online including wiki cannot clearly explain the process
# hence, the good idea would be to read info on wikipedia
# and download an excel spreadsheet made by joeu2004
# formulas are always more straight forward than descriptions

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf


# the calculation of sar
# as rules are very complicated
# plz check the links above to understand more about it

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

def signal_generation(df, method):
    new = method(df)

    new['positions'], new['signals'] = 0, 0
    new['positions'] = np.where(new['real sar'] < new['Close'], 1, 0)
    new['signals'] = new['positions'].diff()

    return new


# plotting of sar and trading positions
# still similar to macd

def plot(new, ticker):
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
    plt.show()

def main():
    # download data via fix yahoo finance library
    stdate = input('Enter start date: ')
    eddate = input('Enter end date: ')
    ticker = input('Enter ticker symbol: ')

    # slice is used for plotting
    # a two year dataset with 500 variables would be too much for a figure
    slicer = 450

    df = yf.download(ticker, start=stdate, end=eddate)

    # delete adj close and volume
    # as we dont need them
    del df['Adj Close']
    del df['Volume']

    # no need to iterate over timestamp index
    df.reset_index(inplace=True)

    new = signal_generation(df, parabolic_sar)
    print(new.head())
    # convert back to time series for plotting
    # so that we get a date x axis
    new.set_index(new['Date'], inplace=True)

    # shorten our plotting horizon and plot
    new = new[slicer:]
    plot(new, ticker)


if __name__ == '__main__':
    main()










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










