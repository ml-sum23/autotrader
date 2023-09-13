import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from src.models import *

def predict(args,main_data):
  #read data
  assert 'Date' in main_data.columns, "date column not in given training data"
  main_data.set_index('Date', inplace = True)
  print(f'There are {main_data.shape[0]} lines of data in total, starting {main_data.index[0]}.')

  #check if there is enough data
  if args.sequential:
    assert args.training_size + args.window_size + 31 < main_data.shape[0], "not enough data"
  else:
    assert args.training_size + 31 < main_data.shape[0], "not enough data"

  #return rate normalization
  if args.return_normalization:
    return_data = main_data[(main_data != 0).all(axis=1)].pct_change()
    return_data.dropna(inplace = True)
    data_backtesing = return_data.copy()
  else:
    data_backtesing = main_data.copy()

  if args.min_max_normalization:
    scaler = MinMaxScaler()
    data_backtesing = scaler.fit_transform(data_backtesing)

  if args.standard_normalization:
    scaler = StandardScaler()
    data_backtesing = scaler.fit_transform(data_backtesing)

  #final data holder
  prediction_list = []

  if args.sequential:
    testing_size = main_data.shape[0] - args.training_size - args.window_size
  else:
    testing_size = main_data.shape[0] - args.training_size

  for day in tqdm(range(testing_size)):
    #sequential data processing
    if args.sequential:
      data_current_epoch = []
      #creat rolling window for sequential model
      for window in range(args.training_size + 1):
        sliced = data_backtesing[day + window:day+window+args.window_size]
        data_current_epoch.append(sliced)
      #final data
      data_current_epoch = np.stack(data_current_epoch, axis = 0)
      X_train = data_current_epoch[:-1,:,:-1]
      X_test = np.expand_dims(data_current_epoch[-1,:,:-1], axis = 0)
      y_train = data_current_epoch[:-1,:,-1][:,-1]
      #y_test = data_current_epoch[-1,:,-1][-1]
    #non-sequential data processing
    else:
      X_train = data_backtesing[day:day+args.training_size, :-1]
      X_test = data_backtesing[day+args.training_size:day+args.training_size+1, :-1]
      y_train = data_backtesing[day:day+args.training_size, -1]
      #y_test = data_backtesing[day+args.training_size:day+args.training_size+1, -1]

    prediction = eval(f'{args.model}(X_train,y_train,X_test,scaler)')
    prediction_list.append(prediction)

  main_data['prediction'] = [np.nan for _ in range(main_data.shape[0] - len(prediction_list))] + prediction_list
  return main_data.dropna(subset = ['prediction'])
