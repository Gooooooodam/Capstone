import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# simple moving average
def macd(signals):
    signals['ma1'] = signals['Close'].rolling(window=ma1, min_periods=1, center=False).mean()
    signals['ma2'] = signals['Close'].rolling(window=ma2, min_periods=1, center=False).mean()

    return signals

def signal_generation(df, method):
    signals = method(df)
    signals['positions'] = 0

    # positions becomes and stays one once the short moving average is above long moving average
    signals['positions'][ma1:] = np.where(signals['ma1'][ma1:] >= signals['ma2'][ma1:], 1, 0)

    # as positions only imply the holding
    # we take the difference to generate real trade signal
    signals['signals'] = signals['positions'].diff()

    # oscillator is the difference between two moving average
    # when it is positive, we long, vice versa
    signals['oscillator'] = signals['ma1'] - signals['ma2']

    return signals


# plotting the backtesting result
def plot(new, ticker):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    new['Close'].plot(label=ticker)
    ax.plot(new.loc[new['signals'] == 1].index, new['Close'][new['signals'] == 1], label='LONG', lw=0, marker='^',
            c='g')
    ax.plot(new.loc[new['signals'] == -1].index, new['Close'][new['signals'] == -1], label='SHORT', lw=0, marker='v',
            c='r')

    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Positions')

    plt.show()

    fig = plt.figure()
    cx = fig.add_subplot(211)

    new['oscillator'].plot(kind='bar', color='r')

    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks([])
    plt.xlabel('')
    plt.title('MACD Oscillator')

    bx = fig.add_subplot(212)

    new['ma1'].plot(label='ma1')
    new['ma2'].plot(label='ma2', linestyle=':')

    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def main():

    global ma1, ma2, stdate, eddate, ticker, slicer

    ma1 = 10
    ma2 = 21
    stdate = input('start date in format yyyy-mm-dd:')
    eddate = input('end date in format yyyy-mm-dd:')
    ticker = input('ticker:')

    slicer = int(input('slicing:'))

    # downloading data
    df = yf.download(ticker, start=stdate, end=eddate)

    new = signal_generation(df, macd)
    new = new[slicer:]
    plot(new, ticker)

if __name__ == '__main__':
    main()