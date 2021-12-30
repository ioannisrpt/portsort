# PortSort
The PortSort class enables single, double or triple sorting of entities in portfolios. Aggregation on a variable of interest is possible
in the single sort regime only. The double and triple sorts can be conditional or unconditional. Construction of portfolios based on firm characteristics
is now possible with the FFPortfolios() method.

# How to install 
```python
pip install PortSort
```

# How to use ...
The PortSort_Example.py file is used to demonstrate the functionality of the PortSort class and its methods. 

A panel dataset of the characteristics of 800 random US domiciled and traded securities for 2018-2020 and a 
dataset of monthly returns are used for the example. 

The 'FirmCharacteristics2018.csv' dataset has 8 columns:
1. 'year' : Calendar year
2. 'notPERMNO' : Randomized security identifier based on the true CRSP PERMNO
3. 'CAP' : Market Capitalization of the last business day of the current year
4. 'CAP_W' :  Market Capitalization of the last business day of the previous year
5. 'RET_total' : Total return (including dividends) of a stock in a year
6. 'SPREAD_PC_median' : Daily median of the ratio of the Bid-Ask spread over the closing price of the stock
                      in a year
7. 'FF30' : Fama-French 30 industry classification of a security based on its SIC.
8. 'EXCHCD' : Market exchange in which the security is traded. 1 is NYSE, 2 is AMEX and 3 is NASDAQ.

The 'STOCKmonthlydata2019.csv' dataset has 4 columns:
1. 'date_m': Date in YYYYmm format.
2. 'year' : Calendar year
3. 'RET' : Returns of securities in monthly frequency (not in percentage).
4. 'notPERMNO' : Randomized security identifier based on the true CRSP PERMNO


```python

"""
Examples of the PortSort class and its methods.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Main directory
wdir = r'C:\Users\ropot\OneDrive\Desktop\Python Scripts\Update PortSort'
os.chdir(wdir)


# Import the PortSort class. For more details: 
# https://github.com/ioannisrpt/PortSort.git 
# pip install PortSort
from PortSort import PortSort as ps

# Import FirmCharacteristics table (annual frequency)
ftotype32 = {'year' : np.int32, 'RET_total':np.float32, \
             'SPREAD_PC_median':np.float32, 'EXCHCD':np.int32, 'notPERMNO':np.int32}
firmchars = pd.read_csv(os.path.join(wdir, 'FirmCharacteristics2018.csv')).astype(ftotype32)

# Import return data (monthly frequency)
ctotype32 = {'RET':np.float32, 'date_m':np.int32, 'year':np.int32, 'notPERMNO':np.int32}
crspm = pd.read_csv(os.path.join(wdir, 'STOCKmonthlydata2019.csv')).astype(ctotype32)


# Define the PortSort class
portchar = ps.PortSort(df = firmchars, entity_id = 'notPERMNO', time_id = 'year', \
                    save_dir = wdir)


# ------------
# SingleSort()
# ------------

# Single Sort stocks into quintile portfolios based on  the market capitalization 
# of the last business day of the previous year ('CAP')
portchar.SingleSort(firm_characteristic = 'CAP', lagged_periods = 1, n_portfolios = 5)
print(portchar.single_sorted.head(20))


# Single Sort stocks into 3 portfolios (30%, 40% 30%) based on the market capitalization 
# of the last business day of the previous year ('CAP').
# NYSE breakpoints for size are used.
portchar.SingleSort(firm_characteristic = 'CAP',  lagged_periods = 1, \
                    n_portfolios = np.array([0, 0.3, 0.7]), \
                    quantile_filter = ['EXCHCD', 1])
print(portchar.single_sorted.head(20))


# ------------
# DoubleSort()
# ------------

# Double Sort stocks unconditionally into 5x2 portfolios based on the market capitalization 
# of the last business day of the previous year ('CAP') and the total annual return 
# of the past year ('RET_total').
portchar.DoubleSort(firm_characteristics = ['CAP', 'RET_total'],  lagged_periods = [1,1], \
                    n_portfolios = [5,2])
print(portchar.double_sorted.head(20))


# Double Sort stocks conditionally into 5x2 portfolios based on  the market capitalization 
# of the last business day of the previous year ('CAP') and the total annual return 
# of the past year ('RET_total').
# NYSE breakpoints for size are used.
portchar.DoubleSort(firm_characteristics = ['CAP', 'RET_total'], lagged_periods = [1,1], \
                    n_portfolios = [5,2], quantile_filters = [['EXCHCD', 1], None], \
                    conditional = True)
print(portchar.double_sorted.head(20))



# ------------
# TripleSort()
# ------------

# Triple Sort stocks unconditionally into 2x2x2 portfolios based on  the market capitalization 
# of the last business day of the previous year ('CAP'), total annual return ('RET_total')
# and daily median spread percentage ('SPREAD_PC_median') of the past year.
# NYSE breakpoitns are used for size and spread percentage but not for total return.
portchar.TripleSort(firm_characteristics = ['CAP', 'RET_total', 'SPREAD_PC_median'], \
                    lagged_periods = [1,1,1], n_portfolios = [2,2,2], \
                    quantile_filters = [['EXCHCD', 1], None, ['EXCHCD', 1]])
print(portchar.triple_sorted.head(20))


# --------------
# FFPortfolios()
# --------------


# Monthly returns of 10 value-weighted portfolios on size ('CAP'). NYSE breakpoints are used.
portchar.FFPortfolios(ret_data = crspm, ret_time_id = 'date_m', FFcharacteristics = ['CAP'], \
                      FFlagged_periods = [1], FFn_portfolios = [5], \
                      FFquantile_filters = [['EXCHCD',1]], weight_col = 'CAP_W', \
                      return_col = 'RET', FFsave = True)
print(portchar.FFportfolios.head(30))
print(portchar.FFnum_stocks)


# Monthly returns of 5x2 value-weighted portfolios on size ('CAP') and liquidity ('SPREAD_PC_median').
# The sort is unconditional and NYSE breakpoints are used for size.
portchar.FFPortfolios(ret_data = crspm, ret_time_id = 'date_m', \
                      FFcharacteristics = ['CAP', 'SPREAD_PC_median'], \
                      FFlagged_periods = [1,1], FFn_portfolios = [5,2], \
                      FFquantile_filters = [['EXCHCD',1], None], FFconditional = False, \
                      weight_col = 'CAP_W', return_col = 'RET', FFsave = True)
print(portchar.FFportfolios.head(30))
print(portchar.FFnum_stocks)



    
# Monthly returns of 2x2x2 value-weighted portfolios on size ('CAP'),liquidity ('SPREAD_PC_median') and 
# annual returns ('RET_total') of the previous year.
# The sorts are conditional (the order matters) and NYSE breakpoints are used for size.
portchar.FFPortfolios(ret_data = crspm, ret_time_id = 'date_m', \
                      FFcharacteristics = ['CAP', 'SPREAD_PC_median', 'RET_total'], \
                      FFlagged_periods = [1,1,1], FFn_portfolios = [2,2,2], \
                      FFquantile_filters = [['EXCHCD',1], None, None], FFconditional = True, \
                      weight_col = 'CAP_W', return_col = 'RET', FFsave = True)
print(portchar.FFportfolios.head(30))
print(portchar.FFnum_stocks)    

```









