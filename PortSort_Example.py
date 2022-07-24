# -*- coding: utf-8 -*-
# Python 3.7.7
# Pandas 1.0.5
# Author: Ioannis Ropotos

"""
Examples of the PortSort class and its methods.
"""

import os
import pandas as pd
import numpy as np



# Main directory
wdir = r'C:\Users\ropot\Desktop\portsort_testing'
os.chdir(wdir)


# Import the PortSort class. For more details: 
# https://github.com/ioannisrpt/portsort.git 
# pip install portsort 
from portsort import portsort as ps


# Import FirmCharacteristics table (annual frequency)
ftotype32 = {'year' : np.int32, 
             'CAP' : np.float32,
             'CAP_W' : np.float32,
             'RET_total' : np.float32, 
             'SPREAD_PC_median' : np.float32, 
             'EXCHCD' : np.int32,
             'notPERMNO' : np.int32}
firmchars = pd.read_csv(os.path.join(wdir, 'FirmCharacteristics2018.csv')).astype(ftotype32)

# Import return data (monthly frequency)
ctotype32 = {'RET' : np.float32,
             'date_m' : np.int32, 
             'year' : np.int32, 
             'notPERMNO' : np.int32}
crspm = pd.read_csv(os.path.join(wdir, 'STOCKmonthlydata2019.csv')).astype(ctotype32)


# Define the PortSort class
portchar = ps.PortSort(df = firmchars, 
                      entity_id = 'notPERMNO', 
                      time_id = 'year', 
                      save_dir = wdir)



# -------------
# single_sort()
# -------------

# Single sort stocks into quintile portfolios based on  the market 
# capitalization of the last business day of the previous year ('CAP')
portchar.single_sort(firm_characteristic = 'CAP', 
                    lagged_periods = 1, 
                    n_portfolios = 5)
print(portchar.single_sorted.head(20))


# Single sort stocks into 3 portfolios (30%, 40% 30%) based on the market 
# capitalization of the last business day of the previous year ('CAP').
# NYSE breakpoints for size are used.
portchar.single_sort(firm_characteristic = 'CAP', 
                    lagged_periods = 1, 
                    n_portfolios = np.array([0, 0.3, 0.7]), 
                    quantile_filter = ['EXCHCD', 1])
print(portchar.single_sorted.head(20))


# -------------
# double_sort()
# -------------

# Double sort stocks unconditionally into 5x2 portfolios based on the market
# capitalization of the last business day of the previous year ('CAP') and the 
# total annual return of the past year ('RET_total').
portchar.double_sort(firm_characteristics = ['CAP', 'RET_total'], 
                    lagged_periods = [1,1], 
                    n_portfolios = [5,2])
print(portchar.double_sorted.head(20))


# Double sort stocks conditionally into 3x2 portfolios based on  the market 
# capitalization of the last business day of the previous year ('CAP') and the
# total annual return of the past year ('RET_total').
# NYSE breakpoints for size are used.
portchar.double_sort(firm_characteristics = ['CAP', 'RET_total'], 
                    lagged_periods = [1,1], 
                    n_portfolios = [np.array([0, 0.3, 0.7]), 2],
                    quantile_filters = [['EXCHCD', 1], None], 
                    conditional = True)
print(portchar.double_sorted.head(20))



# -------------
# triple_sort()
# -------------

# Triple Sort stocks unconditionally into 2x2x2 portfolios based on  the market
# capitalization of the last business day of the previous year ('CAP'), total 
# annual return ('RET_total') and daily median spread percentage 
# ('SPREAD_PC_median') of the past year.
# NYSE breakpoitns are used for size and spread percentage 
# but not for total return.
portchar.triple_sort(firm_characteristics=['CAP', 'RET_total', 'SPREAD_PC_median'], 
                    lagged_periods = [1,1,1], 
                    n_portfolios = [2,2,2], 
                    quantile_filters = [['EXCHCD', 1], None, ['EXCHCD', 1]])
print(portchar.triple_sorted.head(20))


# Triple Sort stocks into 2x2x2 portfolios based on  the market capitalization 
# of the last business day of the previous year ('CAP'), total annual return 
# ('RET_total') and daily median spread percentage ('SPREAD_PC_median') of the 
# past year.
# First stocks are uncondtionally sorted by size and total annual return and 
# then within these portfolios they are conditionally sorted by spread.
# If A, B, C are the characteristics in that order and '+', '|' correspond
# to intersection and conditionality of sets, then conditional = [False, True] 
# is equivalent to  C|(A+B).
# Type help(PortSort.triple_sort) for more details.
# NYSE breakpoitns are used for size and spread percentage 
# but not for total return.
portchar.triple_sort(firm_characteristics=['CAP', 'RET_total', 'SPREAD_PC_median'], 
                    lagged_periods = [1,1,1], 
                    n_portfolios = [2,2,2], 
                    quantile_filters = [['EXCHCD', 1], None, ['EXCHCD', 1]],
                    conditional = [False, True])
print(portchar.triple_sorted.head(20))


# Triple Sort stocks into 2x2x2 portfolios based on  the market capitalization 
# of the last business day of the previous year ('CAP'), total annual return 
# ('RET_total') and daily median spread percentage ('SPREAD_PC_median') of the
# past year. Entities conditional on size, are then sorted into 2x2 
# unconditional return and spread portfolios. 
# If A, B, C are the characteristics in that exact order and
# '+', '|' correspond to intersection and conditionality of sets, 
# then conditional = [True, False] is  equivalent to (B + C)| A. 
# Type help(PortSort.triple_sort) for more details.
# NYSE breakpoitns are used for size and spread percentage 
# but not for total return.
portchar.triple_sort(firm_characteristics=['CAP', 'RET_total', 'SPREAD_PC_median'], 
                    lagged_periods = [1,1,1], 
                    n_portfolios = [2,2,2], 
                    quantile_filters = [['EXCHCD', 1], None, ['EXCHCD', 1]],
                    conditional = [True, False])
print(portchar.triple_sorted.head(20))


# ---------------------
# augment_last_traded()
# ---------------------


# First we adjust for delisted returns during the calendar
# year. PortSort handles the firm characteristics and the
# return dataframe separately and only merge them together
# at the very end for the calculation of portfolio returns.
# As such, we need to augment the characteristics dataset
# with the data for stocks that are delisted but need to be
# included in the sorting procedure. augment_last_traded()
# method allows for that adjustment while it fills the extra
# rows with the weighting variable 'CAP_W' and the exchange
# market 'EXCHCD'. If we don't adjust for the delistings,
# our results will suffer from look-ahead bias.
portchar.augment_last_traded(ret_data = crspm,
                            ret_time_id = 'date_m',
                            col_w='CAP',
                            col_w_lagged_periods=1,
                            col_w_suffix = 'W',
                            fill_cols=['EXCHCD'])
                            
# ---------------
# ff_portfolios()
# ---------------

# Monthly returns of 10 value-weighted portfolios on size ('CAP'). 
# NYSE breakpoints are used.
portchar.ff_portfolios(ret_data = crspm,
                      ret_time_id = 'date_m',
                      ff_characteristics = ['CAP'], 
                      ff_lagged_periods = [1], 
                      ff_n_portfolios = [10], 
                      ff_quantile_filters = [['EXCHCD',1]], 
                      weight_col = 'CAP_W', 
                      return_col = 'RET', 
                      ff_save = True)
print(portchar.portfolios.head(30))
print(portchar.num_stocks)


# Monthly returns of 3x2 value-weighted portfolios on size ('CAP') and 
# liquidity ('SPREAD_PC_median').
# The sort is unconditional and NYSE breakpoints are used for size. 
# By specifying the market_cap_cols, the portfolio turnover is also returned.
# market_cap_cols is a list =
# [capitalization of the stock at the end of the previous period, 
# capitalization of the stock at the ned of the current period]
portchar.ff_portfolios(ret_data = crspm, 
                      ret_time_id = 'date_m', 
                      ff_characteristics = ['CAP', 'SPREAD_PC_median'], 
                      ff_lagged_periods = [1, 1],
                      ff_n_portfolios = [np.array([0, 0.3, 0.7]),2], 
                      ff_quantile_filters = [['EXCHCD',1], None],
                      ff_conditional = [False], 
                      weight_col = 'CAP_W', 
                      return_col = 'RET', 
                      market_cap_cols = ['CAP_W', 'CAP'],
                      ff_save = True)
print(portchar.portfolios.head(30))
print(portchar.num_stocks)
print(portchar.turnover)
print('Acess the explicit portfolio weights of the stocks: \n')
print(portchar.turnover_raw.head(20))



    
# Monthly returns of 2x2x2 value-weighted portfolios on size ('CAP'), 
# liquidity ('SPREAD_PC_median') and annual returns ('RET_total') of the
# previous year. 
# The sorts are all conditional (the order matters).
# NYSE breakpoints are used only for size.
portchar.ff_portfolios(ret_data = crspm,
                      ret_time_id = 'date_m', 
                      ff_characteristics = ['CAP', 'SPREAD_PC_median', 'RET_total'], 
                      ff_lagged_periods = [1,1,1], 
                      ff_n_portfolios = [2,2,2], 
                      ff_quantile_filters = [['EXCHCD',1], None, None],
                      ff_conditional = [True, True], 
                      weight_col = 'CAP_W', 
                      return_col = 'RET', 
                      market_cap_cols = ['CAP_W', 'CAP'],
                      ff_save = True)
print(portchar.portfolios.head(30))
print(portchar.num_stocks)    
print(portchar.turnover)




# Monthly returns of 2x2x2 value-weighted portfolios on size ('CAP'),
# liquidity ('SPREAD_PC_median') and annual returns ('RET_total') of the 
# previous year.
# Entities conditional on size, are then sorted into 2x2 unconditional return 
# and spread portfolios. 
# If A, B, C are the characteristics in that exact order and
# '+', '|' correspond to intersection and conditionality of sets, 
# then conditional = [True, False] is  equivalent to (B + C)| A. 
# Type help(PortSort.ff_portfolios) for more details.
# NYSE breakpoints are used for size.
portchar.ff_portfolios(ret_data = crspm,
                      ret_time_id = 'date_m', 
                      ff_characteristics=['CAP', 'SPREAD_PC_median', 'RET_total'], 
                      ff_lagged_periods = [1,1,1], 
                      ff_n_portfolios = [2,2,2], 
                      ff_quantile_filters = [['EXCHCD',1], None, None],
                      ff_conditional = [True, False], 
                      weight_col = 'CAP_W', 
                      return_col = 'RET', 
                      market_cap_cols = ['CAP_W', 'CAP'],
                      ff_save = True)
print(portchar.portfolios.head(30))
print(portchar.num_stocks)    
print(portchar.turnover)







