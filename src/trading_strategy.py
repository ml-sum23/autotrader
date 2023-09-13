import pandas as pd
import numpy as np
from scipy import stats
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def calculate_final_accum_return(lsdr, bhdr, lsar, bhar,length, k):
    '''with a given parameter k, conduct linear regression and gives the final accumulate return
    '''
    final_accum_return = 1
    flex_strat = np.nan
    daily_return = 0

    signal_list = [1/2*(lsar[i]+bhar[i]) for i in range(length)]
    for j in range(1,length):
        signal_0 = signal_list[j-2]
        signal_1 = signal_list[j-1]
        x = list(range(max(j - k, 0), j))
        y = signal_list[max(j - k, 0):j]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        criteria_1 = slope * (j - 1) + intercept
        criteria_0 = slope * j + intercept

        if signal_1 <= criteria_1 and signal_0 > criteria_0: #go up
            flex_strat = 'ls' if lsar[j-1] >= bhar[j-1] else 'bh'
        elif signal_1 >= criteria_1 and signal_0 < criteria_0: #go down
            flex_strat = 'ls' if lsar[j-1] <= bhar[j-1] else 'bh'
        else: #no change
              flex_strat = flex_strat
        
        if flex_strat == 'ls':
            daily_return = lsdr[j]
        elif flex_strat == 'bh':
            daily_return = bhdr[j]
        else: #np.nan
            daily_return = 0
        
        final_accum_return *= 1 + daily_return

    return final_accum_return, flex_strat



def optimize_k(lsdr, bhdr, lsar, bhar,length):
    '''find the best sample size for linear regression to maximize final accumulate return
    '''
    best_parameter_value = None
    best_accum_rate = 0

    for parameter_value in range(1, 100):  # You can adjust the range of parameter values to search
        accum_rate = calculate_final_accum_return(lsdr, bhdr, lsar, bhar,length=length, k=parameter_value)[0]
        if accum_rate > best_accum_rate:
            best_accum_rate = accum_rate
            best_parameter_value = parameter_value
    return best_parameter_value


def trading_decision(main_data,test_day):
    #final data holder
    long_short_accum_return = []
    long_short_daily_return = []
    long_short_trading_direction = [] #1 for long, -1 for short, 0 for no trading

    buy_hold_accum_return = []
    buy_hold_daily_return = []
    buy_hold_trading_direction = [] #1 for long, -1 for short, 0 for no trading

    #set initials
    ls_accum = bh_accum = 1
    ls_daily = bh_daily = 0
    ls_direction = bh_direction = 1
    flex_strat = flex_direction = np.nan

    for day in tqdm(range(test_day)):
        pred_value = main_data.iloc[day, main_data.columns.get_loc('prediction')]
        prev_value = main_data.iloc[day, main_data.columns.get_loc('Close_0')]
        true_value = main_data.iloc[day, main_data.columns.get_loc('Shift_Close')]

        #long short strategy
        if pred_value > prev_value: #long
            ls_daily = (true_value - prev_value)/prev_value
            ls_direction = 1
        elif pred_value < prev_value: #short
            ls_daily = (prev_value - true_value)/prev_value
            ls_direction = -1
        else: #no trading
            ls_daily = 0
            ls_direction = 0

        ls_accum *= 1 + ls_daily

        long_short_accum_return.append(ls_accum)
        long_short_daily_return.append(ls_daily)
        long_short_trading_direction.append(ls_direction)

        #buy hold strategy
        bh_daily = (true_value - prev_value)/prev_value
        bh_accum *= 1 + bh_daily
        bh_direction = 1

        buy_hold_accum_return.append(bh_accum)
        buy_hold_daily_return.append(bh_daily)
        buy_hold_trading_direction.append(bh_direction)

    #flexible strategy
    #signal.append(1/2 * (ls_accum + bh_accum))

    temp_k = max(optimize_k(lsdr = long_short_daily_return, bhdr = buy_hold_daily_return, lsar = long_short_accum_return, bhar = buy_hold_accum_return, length = test_day-1),2)

    flex_strat = calculate_final_accum_return(lsdr = long_short_daily_return, bhdr = buy_hold_daily_return, lsar = long_short_accum_return, bhar = buy_hold_accum_return, length=test_day, k=temp_k)[1]
    #print(f'decision based on {main_data.index[day-2]} and {main_data.index[day-1]}, flex strat for {main_data.index[day]} is {flex_strat}')
    if flex_strat is np.nan:
        flex_direction = 0
    else:
        flex_direction = ls_direction if  flex_strat == 'ls'  else bh_direction

    return flex_direction
