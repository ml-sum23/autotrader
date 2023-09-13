import datetime as dt
from datetime import datetime, timedelta, date, time
import yfinance as yf
import pandas as pd
import numpy as np
from finta import TA
import ta
import argparse
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pytz

def data_downloader(args):
  if args.start_date:
    main_data = yf.download(args.stock, start = datetime.strptime(args.start_date, '%Y-%m-%d') - timedelta(days=args.room_na),
                            end = args.end_date,
                            interval = args.interval)
  else:
    main_data = yf.download(args.stock,
                            start=str(dt.date.today() - dt.timedelta(days=int(args.days/0.5))),
                            end = args.end_date,
                            interval = args.interval)
    


  # Define the timezone for U.S. Eastern Time
  eastern = pytz.timezone('US/Eastern')
  # Get the current time in UTC and convert it to Eastern Time
  now_eastern = datetime.now(pytz.utc).astimezone(eastern)
  # Extract the hour from the current time
  current_hour = now_eastern.hour
    
  # If the market is not closed for today, update today's close price
  if current_hour > 16: 
      # Market is closed, data already reflects today's close
      pass
  else:
      # Market is open, fetch the current price and update today's row
      current_price = yf.Ticker(args.stock).history(period='1m')
      current_price = current_price[['Open','High','Low','Close']].iloc[0].tolist() + [current_price['Close'].iloc[0]] + [current_price['Volume'].iloc[0]]
      main_data.loc[datetime.now().strftime('%Y-%m-%d')] = current_price

  main_data.dropna(inplace = True)
  main_data.drop_duplicates(inplace = True)

  if args.need_log:
    main_data_log = main_data.applymap(np.log)
    main_data_log.columns = ['log_' + i for i in main_data_log.columns]
    main_data = pd.concat([main_data, main_data_log],axis = 1)
    main_data.dropna(subset= main_data.columns[:-1],inplace = True)
    main_data.drop_duplicates(inplace = True)

  if args.need_ta:
    for indicater in eval(args.choices_ta):
      main_data[indicater] = eval('TA.' + indicater + '(main_data)')

  if args.need_macd:
    for column in eval(args.columns_macd):
      macd_indicator = ta.trend.MACD(main_data[column],
                                    window_slow = args.long_span,
                                    window_fast = args.short_span,
                                    window_sign = args.signal_span)
      main_data['MACD'+ str(column)] = macd_indicator.macd()
      main_data['MACD_Signal'+ str(column)] = macd_indicator.macd_signal()
      main_data['MACD_Histogram'+ str(column)] = macd_indicator.macd_diff()

  if args.need_ema:
    for day in eval(args.days_ema):
      assert day <= int(main_data.shape[0]*0.2), "EMA days specified is too large"
      main_data['EMA'+str(day)] = TA.EMA(main_data,day)

  if args.need_prev:
    for column in eval(args.columns_prev):
      assert args.days_prev <= int(main_data.shape[0]*0.2), "Previous days specified is too large"
      for i in reversed(range(1, args.days_prev + 1)):
        main_data[column + '_' + str(i)] = main_data[column].shift(i)

  if args.shift_target:
    main_data['Close_0'] = main_data['Close']
    main_data['Shift_Close'] = main_data['Close'].shift(-1)
    main_data.drop(columns = ['Close', 'Adj Close'], inplace = True)

  main_data.dropna(subset = main_data.columns[:-1],inplace = True)
  main_data.drop_duplicates(inplace = True)
  
  return main_data.reset_index()