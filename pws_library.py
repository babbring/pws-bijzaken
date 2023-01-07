import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

def get_data(tickers, start, end):  
    prices_df = pd.DataFrame()
    columns_names = []

    for ticker in tickers:
        df = yf.download(ticker, auto_adjust=False, start=start, end=end)
        new_column = df['Close']
        prices_df = pd.concat([prices_df, new_column], axis=1)
        columns_names.append(ticker)
    prices_df.columns = columns_names
    return prices_df

def cointegration_test(ticker1, ticker2, start, end):
    df = get_data([ticker1, ticker2], start=start, end=end)
    
    order1 = {"dep":ticker1, "indep":ticker2}
    order2 = {"dep":ticker2, "indep":ticker1}
    
    for order in [order1, order2]:
        #bereken beta
        y = df[order['dep']].tolist()
        x = df[order['indep']].tolist()

        x = sm.add_constant(x)

        results = sm.OLS(y, x).fit()
        order['beta'] = results.params[1]
        print(f"beta:{order['beta']}")
        order['alpha'] = results.params[0]
        print(f"alpha:{order['alpha']}")
        
        #adfuller test op spread
        spread = df[order['dep']] - order['beta']*df[order['indep']]
        adf = ts.adfuller(spread)
        order['p-value'] = adf[1]
        print(order['p-value'])
        
        #bereken halflife
        df3 = pd.DataFrame(spread, columns=['spread'])
        df3['spread_shift'] = df3['spread'].shift()
        df3['dspread'] = df3['spread'] - df3['spread_shift']
        df3 = df3.dropna()
        
        y3 = df3['dspread'].tolist()
        x3 = df3['spread_shift'].tolist()

        x3 = sm.add_constant(x3)

        results3 = sm.OLS(y3, x3).fit()
        halflife = -np.log(2)/results3.params[1]
        order['halflife'] = halflife
        print(halflife)
        
        
    if order1['p-value'] < 0.05 and order2['p-value'] < 0.05:
        if order1['p-value'] < order2['p-value']:
            return order1
        else:
            return order2
    else:
        print("not both good cointegration")