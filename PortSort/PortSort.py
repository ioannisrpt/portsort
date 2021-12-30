# -*- coding: utf-8 -*-
# Python 3.7.7
# Pandas 1.0.5
# Author: Ioannis Ropotos

"""


Class: 
------
    PortSort
    
    Methods:
    --------
        Sort()         
        SingleSort()     
        DoubleSort()       
        TripleSort()
        SingleSortAgg()
        FFPortfolios()


"""



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         SOME USEFUL FUNCTIONS                         #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def WeightedMean(x, df, weights):
    """
    Define the weighted mean function
    """
    return np.average(x, weights = df.loc[x.index, weights])



def save_df(df, filename, save_dir = os.getcwd()):
    # true filename
    name = filename +'.csv'
    # Full file path
    file_path = os.path.join(save_dir, name)
    # save as csv 
    df.to_csv(file_path)
    
    
# Function that converts any column of a dataframe that conntains the string 'date' 
# to a datetime object. 
# Implicitly, datetime format is inferred from data.
def ConvertDate(df):
    """
    Parameters:
    ------------
    df: dataframe
        input dataframe
        
    Returns:
    --------
    df_new: dataframe
        transformed dataframe
    """
    for name in df.columns:
        if 'date' in name.lower():
            # Conert to datetime object of monthly period
            df[name] = pd.to_datetime(df[name])
    return df





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         PREPARE DATA FOR SORTING PORTFOLIOS (GENERAL CHARACTERISTIC)     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def PrepareForSorting(df, firm_characteristic, n_portfolios = 10, \
                      lagged_periods = 1, entity_id = 'PERMNO', time_id = 'Date', \
                      quantile_filter = None):
    
    
    """
    Parameters:
    -----------
    df : dataframe
        Input dataframe that contains information in a panel format.
    firm_characteristic : string
        Sort portfolios based on firm characteristic. It must exist in the column of df.
    n_portfolios : integer or array, default=10
        If n_portfolios is integer, then quantiles will be calculated so that n_portfolios 
        will be constructed.
        If n_portfolios is an array, then quantiles will be calculated according to the array.
    lagged_periods : integer, Default=1
        The value of the firm characteristic used for sorting would be the value lagged_periods before. 
    entity_id : string, Default='PERMNO'
        This is the column that corresponds to the entities/stocks. 
    time_id : string, Default='Date'
        time_id denotes the time dimension the panel dataset. Default is 'Date'
    quantile_filter : list = [column_name, value]
        quantile_filter is a list with the first element being the name of the column and the second
        being the value for which the quantiles should be calculated only. For example, if
        quantile_filter = ['EXCHCD', 1], then only NYSE stocks will be used for the estimation of 
        quantiles. If it is None, all entities will be used.

        
    Returns:
    ---------
    df_new : dataframe
        The new dataframe contains:
            1. A new column, 'firm_characteristic_end', that is the firm_char of a 
               stock of the previous period/year. Rows that have null values in 
               'firm_characteristic_end' are dropped.
            2. Quantiles of firm-characteristic in order to construct n_portfolios portfolios.
            
            
    """
    
    # For simplicity, we denote the firm characteristic with a different name 
    trait = firm_characteristic
   

    # Define the lagged or lead firm characteristic 
    # ---------------------------------------------
    
    # Name of the new column
    if lagged_periods > 0:
        trait_end = trait + '_lag%d' % lagged_periods
    elif lagged_periods < 0:
        trait_end = trait + '_forw%d' % np.abs(lagged_periods)
    else:
        trait_end = trait
    # PERMNO is used as the firm identifier instead of cusip_8 or cusip_8_valide (no difference)
    df[trait_end] = df.groupby(by = entity_id)[trait].shift(periods = lagged_periods)
    # Drop rows with null values in 'trait_end'. This is the only criterion by which 
    # null values are dropped. Only 'trait_end' matters for sorting.
    df = df.dropna(subset = [trait_end])
    
    
    
    # Quantiles for each year based on trait_end and n_portfolios 
    # -----------------------------------------------------------

    
    # Check if n_portfolios is integer 
    if isinstance(n_portfolios, int):
        q_range = np.arange(start = 0, stop = 1, step = 1/n_portfolios)
    # Else it is array
    else:
        q_range = n_portfolios
        
    # Definition of quantiles 
    # -----------------------
    
    if quantile_filter is None:
        trait_q = df.groupby(by = [time_id])[trait_end].quantile(q_range, interpolation = 'linear')
    else:
        # Apply filter for estimation of quantiles
        fil_col = quantile_filter[0]
        fil_value = quantile_filter[1]
        trait_q = df[df[fil_col] == fil_value].groupby(by = [time_id])[trait_end].quantile(q_range, interpolation = 'linear')
    
    

    return df, trait_q




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           ASSIGN QUANTILE TO A STOCK             #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Function that assigns the quantile portfolio for a particular firm based 
# on the firm-characterestic chosen and the quantiles. 

def AssignValue(one_value, quantiles):
    
    """
    Parameters
    ----------
    one_value : float
        Value of the firm characteristic that is compared with the quantiles 
            so that the firm is assigned to a portfolio.
    quantiles : Series
        A Series that with index = quantiles, values = q-quantiles
        and name = trait_lag%d % lagged_periods

    Returns
    -------
    portfolio : integer
        'portfolio' takes the value of 1,2,3,..,n_portfolios if a firm 
        has a characteristic that lies between the corresponding quantiles. 
            

    """
    
    # Number of portfolios
    n_portfolios = len(quantiles)
    
    # Initialize portfolio assignment
    portfolio = 0
    
    for interval in range(1,n_portfolios):
        if one_value <= quantiles.iloc[interval]:
            portfolio = int(interval) # Avoid portfolios denoted as 2.0 or 1.0 
            break
        # If a stock has not been assigned a portfolio with the previous loop, then
        # it belongs to the maximum quantile.
        if portfolio == 0:
            portfolio = n_portfolios
                
    return portfolio



# Function that takes as input a series (quantiles of a firm characteristic) and assign a value 
# depending where a value belongs.

def AssignQuantile(quantiles, data):
    
    """
    Parameters
    ----------
    quantiles : Series
        A Series that contains the quantiles that define intervals 
        from which the quantile portfolio is assigned.
    data : Series 
        A series that contains the value of the firm characteristic 
        to be sorted on a given date/year.

    Returns
    -------
    data_new : Series
        Returns a series of sorted stocks on portfolios.
    

    """
    
    # Extend to the whole series data
    data_new = data.apply(lambda x: AssignValue(x, quantiles))
    
        
    return data_new




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         ASSIGN THE QUANTILE PORTFOLIO TO A STOCK           #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Function that assigns the quantile of the portfolio for a firm characteristic
def AssignPortfolio(df, df_q, time_id = 'Date', save_dir = os.getcwd()):
    
    """
    Parameters
    ----------
    df : dataframe
        Input dataframe that contains all relevant data in a panel format.
    df_q : dataframe
        Dataframe that contains the quantiles of the firm characteristic 
        for each year that are used to sort firms/stocks into portfolios.
        The firm characteristic is included in the name of df_q; 
        firm characteristic = df_q.name
    time_id : string, Default='Date'
        Time dimension of the panel dataset. 
    save_dir : path directory, Default=os.getcwd()
        Directory where the results are saved.

    Returns
    -------
    df_new: dataframe
        Same dataframe as df but it has:
            1. A new column that denotes the quantile of the 
            firm characteristic in which each stock belongs.

    """
    
    # Retrieve Dates/Years; df_q is a multiindex series by definition
    dates = df_q.index.unique(level = 0)
    
    # Retrieve name of the firm characteristic
    firm_characteristic = df_q.name
    
    # Define the new column in df and populate it with null values
    col_name = firm_characteristic+'_portfolio'
    # Define a copy of df to avoid the CopyWarning Error
    df_c = df.copy()
    df_c[col_name] = np.nan
    
    # Iterate through the dates/years to assign a stock in a portfolio
    for date0 in dates:
        quantiles = df_q[date0]
        # Assign portfolios to all stocks of a given year/date
        date_idx = df_c[time_id] == date0
        df_c.loc[date_idx, col_name] = AssignQuantile(quantiles = quantiles, data = df_c.loc[date_idx, firm_characteristic] )
        
    
    return df_c 



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#               CLASS     PortSort                         #         
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




class PortSort:
    
    # Class Variables
    
    # Instance Variables
    def __init__(self, df, entity_id = None, time_id = None,  prefix_name = None,  save_dir = None):

        
        self.entity_id = entity_id if entity_id is not None else 'PERMCO'
        self.time_id = time_id if time_id is not None else 'Date'
        # Create a copy of the input dataframe df. Sort values of df
        # by ['entity_id', 'time_id'] to ensure that lagging or forwarding
        # a firm_characteristic results in the correct sorting characteristic
        self.df = df.sort_values(by = [self.entity_id, self.time_id]).copy(deep = True)
        self.prefix_name = prefix_name if prefix_name is not None else ''
        self.save_dir = save_dir if save_dir is not None else os.getcwd()



       
        

    # ~~~~~~~~~~~~~~~~~~~
    #        Sort       #
    # ~~~~~~~~~~~~~~~~~~~
        
    # Sort entities by characteristic. 
    def Sort(self, df = None, firm_characteristic = 'CAP', lagged_periods = 1,
             n_portfolios = 10, quantile_filter = None, prefix_name = None, save_sort = False):
        
        """
        Method that sorts entities into n_portfolios based on their firm_characteristic of 
        the previous lagged_periods periods. This is a general method to sort and it will be the 
        basis of more complex sorting procedures. It works with instance variables.
        
    
        Parameters
        ----------
        df : dataframe, Default=None
            Input dataframe in a panel data format. If None, then self.df is used.
        firm_characteristic : str, Defaul='CAP'
            Portfolios are sorted based on firm_characteristic. It must exist on the column of df.
        n_portfolios : int or np.array, Default=10
            Number of portfolios that are going to be constructed. Decile portfolios are 
            constructed by default.
        lagged_periods : int, Default=1
            The value of the firm characteristic used would be the value lagged_periods before. 
            Default is 1 so the characteristic one period before would be used to sort stocks 
            into portfolios.
        quantile_filter : list = [column_name, value]
            quantile_filter is a list with the first element being the name of the column and the second
            being the value for which the quantiles should be calculated only. For example, if
            quantile_filter = ['EXCHCD', 1], then only NYSE stocks will be used for the estimation of 
            quantiles. If it is None, all entities will be used.
        prefix_name : string, Default=''
            The prefix of the name of the new dataframe as saved in save_dir. Default is empty ''.
        save_sort : boolean, Default=False
            If True the returns are saved in save_dir.

    
    
        Returns
        -------
        df_sorted : Dataframe
            Dataframe where stocks have been assigned a portfolio based on firm_characteristic.
            
    
        """
        
        # Variables only inside function
        df = df.copy(deep=True) if df is not None else self.df.copy(deep = True)       
        # Count the number of portfolios 
        if not isinstance(n_portfolios, int):
            num_portfolios = len(n_portfolios)
        else:
            num_portfolios = n_portfolios    
            
        # Define the save_folder
        folder_name = '%d_portfolios_SortedBy_%s' % (num_portfolios, firm_characteristic)
        save_folder = os.path.join(self.save_dir, folder_name)

        
        
        # Prepare the data for sorting and define quantiles based on firm characteristic
        df_trait, trait_q = PrepareForSorting(df, firm_characteristic, n_portfolios, \
                                              lagged_periods, \
                                              self.entity_id, self.time_id, \
                                              quantile_filter) 
        
        # Assign portfolio to each stock in our sample
        df_sorted = AssignPortfolio(df_trait, trait_q, time_id = self.time_id)
        
        if save_sort:
            
            if folder_name not in os.listdir(self.save_dir):
                os.mkdir(save_folder)
            # Save portfolios dataframe
            df_sorted_path = os.path.join(save_folder, '%s_of_%d_portfolios_basedOn_%s.csv' % \
                                      (prefix_name, num_portfolios, firm_characteristic))
            df_sorted.to_csv(df_sorted_path, index = False)
            # Save quantiles dataframe
            trait_q_path = os.path.join(save_folder, '%s_quantiles_of_%d_portfolios.csv' % \
                                    (firm_characteristic, num_portfolios))
            trait_q.to_csv(trait_q_path)
        
       
        return df_sorted

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    #      SingleSort          #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Single Sorting 
    def SingleSort(self, firm_characteristic, lagged_periods = None,  n_portfolios = None, \
                   quantile_filter = None, calibrate_cols = None, save_SingleSort = True):
        """
        Method that sorts entities based on only one characteristic using 
        the Sort() method. 
                
        Attributes:
        -----------
        firm_characteristic : str
            Portfolios are sorted based on a single firm_characteristic. 
        lagged_periods : int, Default=1
            The value of the firm characteristic used would be the value lagged_periods before. 
            Default is 1 so the characteristic one period before would be used to sort stocks 
            into portfolios.
        n_portfolios : int or np.array, Default=10
            Number of portfolios that are going to be constructed. Decile portfolios are 
            constructed by default.
        quantile_filter : list = [column_name, value]
            quantile_filter is a list with the first element being the name of the column and the second
            being the value for which the quantiles should be calculated only. For example, if
            quantile_filter = ['EXCHCD', 1], then only NYSE stocks will be used for the estimation of 
            quantiles. If it is None, all entities will be used.
        calibrate_cols : list, Default=None
            Only entities that have non-null values of calibrate_cols, are sorted based on 
            their firm_characteristics. We restrict the set of characteristics that need to be 
            available for entity, as the intersection of calibrate_cols and firm_characteristics.
        save_SingleSort : boolean, Default=True
            If True, the double_sorted dataframe is saved in save_dir.

        Attributes(Returns):
        ---------------------
        single_sorted : DataFrame
            DataFrame in which entities/stocks have been single sorted.
        
        """        
        

        self.firm_characteristic = firm_characteristic
        self.lagged_periods = lagged_periods if lagged_periods is not None else 1
        # Time Arrow
        if self.lagged_periods > 0:
            self.Tarrow = 'lag'
        elif self.lagged_periods < 0:
            self.Tarrow = 'for'
        else:
            self.Tarrow = ''
        # Portfolio name    
        if np.abs(self.lagged_periods)> 0 :
            self.portfolio = '%s_%s%d_portfolio' % (self.firm_characteristic, self.Tarrow, np.abs(self.lagged_periods) )
        else:
            self.portfolio = '%s_portfolio' % (self.firm_characteristic)
        # Portfolio quantiles for sorting
        self.n_portfolios = n_portfolios if n_portfolios is not None else 10 
        # Number of portfolios if n_portfolios is not an integer
        if not isinstance(self.n_portfolios, int):
            self.num_portfolios = len(self.n_portfolios)
        else:
            self.num_portfolios = self.n_portfolios 
        # There is no default value of quantile_filter
        self.quantile_filter = quantile_filter 
        self.calibrate_cols = calibrate_cols
        # Define the save_folder 
        folder_name = '%d_portfolios_SortedBy_%s' % (self.num_portfolios, self.firm_characteristic)
        self.save_folder = os.path.join(self.save_dir, folder_name)
        if folder_name not in os.listdir(self.save_dir):
            os.mkdir(self.save_folder)


           
        # Lag the single characteristic by lagged_periods so we can apply the Sort() function 
        # without lagged periods. Account for calibrate_cols
        # -----------------------------------------------------------------------------------
        
        # Name of the lagged firm characteristic 
        if np.abs(self.lagged_periods) > 0:
            firm_char1 = '%s_%s%d' %  (self.firm_characteristic,   self.Tarrow,   self.lagged_periods)
        else:
            firm_char1 = self.firm_characteristic
        # As a list
        firm_chars = [firm_char1]
        # Create a copy of the original dataframe to be used for double sorting (_ss stands for single sorting) 
        df_ss = self.df.copy()
        # Apply the lagged_periods operator for firm_characteristic
        df_ss[firm_char1] = df_ss.groupby(self.entity_id)[self.firm_characteristic].apply(lambda x: x.shift(self.lagged_periods))
       
        
        # Drop rows for null values in firm_chars and calibrate_cols
        # Dataframe to be used in single sorting
        if self.calibrate_cols is not None:
            df_ss = df_ss.dropna(subset = firm_chars + self.calibrate_cols)
        else:
            df_ss = df_ss.dropna(subset = firm_chars)
    
        # Sort 
        self.single_sorted = self.Sort(df = df_ss, firm_characteristic = firm_char1, n_portfolios = self.n_portfolios, \
                                       lagged_periods = 0, quantile_filter = self.quantile_filter,\
                                        prefix_name = self.prefix_name, save_sort = save_SingleSort).reset_index(drop  = True)
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~
    #      DoubleSort      #
    # ~~~~~~~~~~~~~~~~~~~~~~
        
    # Double Sorting
    def DoubleSort(self, firm_characteristics, lagged_periods = None, n_portfolios = None, \
                   quantile_filters = [None, None], conditional = None, calibrate_cols = None, \
                   save_DoubleSort = True):
        
        """
        Method that sorts entities based on two characteristics. The sort can be 
        unconditional (the 2 sorts are independent of each other) or conditional
        (the second sort is dependent on the first sort).
        
        Attributes:
        -----------
        firm_characteristics : list
            First element = first firm characteristic
            Second element = second firm characteristic
        lagged_periods : list
            First element = lagged periods for first characteristic
            Second element = lagged periods for second characteristic
        n_portfolios : list
            First element = portfolios for sorting on first characteristic (int or np.array)
            Second element = portfolios for sorting on second characteristic (int or np.array)
        quantile_filters : list
            First element = quantile filter for first characteristic
            Second element = quantile filter for second characteristic            
        conditional : boolean, Default=False
            If True, the second sort is conditional on the first. 
            If False, the sorts are indepedent. 
        calibrate_cols : list, Default=None
            Only entities that have non-null values of calibrate_cols, are sorted based on 
            their firm_characteristics. We restrict the set of characteristics that need to be 
            available for entity, as the intersection of calibrate_cols and firm_characteristics.
        save_DoubleSort : boolean, Default=True
            If True, the double_sorted dataframe is saved.


        Attributes(Returns):
        ---------------------
        double_sorted : DataFrame
            DataFrame in which entities/stocks have been double sorted (conditionally or unconditionally).
        
        """
        
        
        
        # First firm characteristic
        # --------------------------
        self.firm_characteristic = firm_characteristics[0] 
        self.lagged_periods = lagged_periods[0]
        # Time arrow
        if self.lagged_periods > 0:
            self.Tarrow = 'lag'
        elif self.lagged_periods < 0:
            self.Tarrow = 'for'
        else:
            self.Tarrow_2 = ''
        # Portfolio name    
        if np.abs(self.lagged_periods)> 0 :
            self.portfolio = '%s_%s%d_portfolio' % (self.firm_characteristic, self.Tarrow, np.abs(self.lagged_periods) )
        else:
            self.portfolio = '%s_portfolio' % (self.firm_characteristic)
        # Portfolio quantiles for sorting   
        self.n_portfolios = n_portfolios[0] 
        # Number of portfolios if n_portfolios is not an integer
        if not isinstance(self.n_portfolios, int):
            self.num_portfolios = len(self.n_portfolios)
        else:
            self.num_portfolios = self.n_portfolios 
        # Filter quantiles 
        self.quantile_filter = quantile_filters[0]
        
        
        # Second firm characteristic
        # --------------------------
        self.firm_characteristic_2 = firm_characteristics[1]
        self.lagged_periods_2 = lagged_periods[1] 
        # Time Arrow 2
        if self.lagged_periods_2 > 0:
            self.Tarrow_2 = 'lag'
        elif self.lagged_periods_2 < 0:
            self.Tarrow_2 = 'for'
        else:
            self.Tarrow_2 = ''
        # Portfolio Name 2
        if np.abs(self.lagged_periods_2)>0:
            self.portfolio_2 = '%s_%s%d_portfolio' % (self.firm_characteristic_2, self.Tarrow_2, np.abs(self.lagged_periods_2) )
        else:
            self.portfolio_2 = '%s_portfolio' % (self.firm_characteristic_2)
        # Portfolio quantiles 2 for sorting
        self.n_portfolios_2 = n_portfolios[1] 
        # Number of portfolios if n_portfolios_2 is not an integer
        if not isinstance(self.n_portfolios_2, int):
            self.num_portfolios_2 = len(self.n_portfolios_2)
        else:
            self.num_portfolios_2 = self.n_portfolios_2
        # Filter quantiles 2
        self.quantile_filter_2 = quantile_filters[1]
        
        
        self.calibrate_cols = calibrate_cols        
        
        # Conditional or unconditional sorting.
        self.conditional = conditional if conditional is not None else False
        # Define the save_folder 
        folder_name = '%dx%d_portfolios_SortedBy_%sand%s' % (self.num_portfolios, self.num_portfolios_2, \
                                                               self.firm_characteristic, self.firm_characteristic_2)
        self.save_folder = os.path.join(self.save_dir, folder_name)
        if folder_name not in os.listdir(self.save_dir):
            os.mkdir(self.save_folder)
        # Save results
        self.save_DoubleSort = save_DoubleSort 
        
        # Function that defines the double sort portfolio column
        def join_func(a,b):
            return '_'.join([str(int(a)), str(int(b))])
        
        
        # Lag the two characteristics by lagged_periods so we can apply the Sort() function 
        # without lagged periods. Account for calibrate_cols
        # -----------------------------------------------------------------------------------
        
        # Name of the lagged firm characteristic columns
        if np.abs(self.lagged_periods) > 0:
            firm_char1 = '%s_%s%d' %  (self.firm_characteristic,   self.Tarrow,   self.lagged_periods)
        else:
            firm_char1 = self.firm_characteristic
        
        if np.abs(self.lagged_periods_2) > 0:
            firm_char2 = '%s_%s%d' %  (self.firm_characteristic_2, self.Tarrow_2, self.lagged_periods_2)
        else:
            firm_char2 = self.firm_characteristic_2
        # As a list
        firm_chars = [firm_char1, firm_char2]
        # Create a copy of the original dataframe to be used for double sorting (_ds stands for double sorting) 
        df_ds = self.df.copy()
        # Apply the lagged_periods operator for firm_characteristic
        df_ds[firm_char1] = df_ds.groupby(self.entity_id)[self.firm_characteristic].apply(lambda x: x.shift(self.lagged_periods))
        # Apply the lagged_periods_2 operator for firm_characteristic_2
        df_ds[firm_char2] = df_ds.groupby(self.entity_id)[self.firm_characteristic_2].apply(lambda x: x.shift(self.lagged_periods_2))
        
        # Drop rows for null values in firm_chars and calibrate_cols
        # Dataframe to be used in double sorting
        if self.calibrate_cols is not None:
            df_ds = df_ds.dropna(subset = firm_chars + self.calibrate_cols)
        else:
            df_ds = df_ds.dropna(subset = firm_chars)
        

        
        if self.conditional:
                                    
            # Single sort on firm_char1 with lagged_periods = 0 on df_ds
            single_sorted = self.Sort(df_ds, firm_characteristic = firm_char1, lagged_periods = 0, \
                                      n_portfolios = self.n_portfolios, quantile_filter = self.quantile_filter, \
                                      save_sort = False)

            # Double sort on firm_char2 with lagged periods = 0 on single_sorted dataframe
            double_sorted = single_sorted.groupby(self.portfolio).apply(lambda x:  self.Sort(x, \
                            firm_characteristic = firm_char2, lagged_periods = 0, n_portfolios = self.n_portfolios_2, \
                            quantile_filter = self.quantile_filter_2, save_sort = False) )
            # Reset the index
            double_sorted.reset_index(drop = True, inplace = True)
                      

            # Define the double sort portfolio column
            double_sorted['Double_sort_portfolio'] = double_sorted.apply(lambda x: join_func(x[self.portfolio], x[self.portfolio_2]), axis = 1)
            
            # Define double_sorted and sort again by entity and time.
            self.double_sorted = double_sorted.sort_values(by = [self.entity_id, self.time_id])
            
            if self.save_DoubleSort:
                filename = 'conditional_double_sort_on_%sand%s.csv' % (self.firm_characteristic, self.firm_characteristic_2)
                self.double_sorted.to_csv(os.path.join(self.save_folder, filename), index = False)

        else:
            
            # Apply self.Sort() two times in a row (nested functions)
            double_sorted = self.Sort( self.Sort(df_ds, firm_char1, 0, self.n_portfolios, self.quantile_filter, \
                                                 save_sort = False), \
                                      firm_char2, 0, self.n_portfolios_2, self.quantile_filter_2, save_sort = False)
                
            # Define the double sort portfolio column
            double_sorted['Double_sort_portfolio'] = double_sorted.apply(lambda x: join_func(x[self.portfolio], x[self.portfolio_2]), axis = 1)
            
            # Define double_sorted and sort again by entity and time.
            self.double_sorted = double_sorted.sort_values(by = [self.entity_id, self.time_id]).reset_index(drop = True)
                        
            
            if self.save_DoubleSort:
                filename = 'unconditional_double_sort_on_%sand%s.csv' % (self.firm_characteristic, self.firm_characteristic_2)
                self.double_sorted.to_csv(os.path.join(self.save_folder, filename), index = False)
                

                

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #       TripleSort           #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    # Triple Sorting
    def TripleSort(self, firm_characteristics, lagged_periods, n_portfolios, quantile_filters = [None, None, None], \
                   conditional = None, calibrate_cols = None, save_TripleSort = True):
        
        """
        Method that sorts entities based on three characteristics. The sort can be 
        unconditional (the 3 sorts are independent of each other) or conditional
        (each sort is dependent on the previous sort).
        
        Attributes:
        -----------
        firm_characteristics : list
           List elements are firm characteristics.
        lagged_periods : list
            List elements are the lagged perios of the firm characteristics.
        n_portfolios : list
            List elements are the portfolio quantiles for sorting (int or np.array)
        quantile_filters : list
            List elements are the quantile filters for the characteristics.           
        conditional : boolean, Default=False
            If True, the second sort is conditional on the first. 
            If False, the sorts are indepedent. 
        calibrate_cols : list, Default=None
            Only entities that have non-null values of calibrate_cols, are sorted based on 
            their firm_characteristics. We restrict the set of characteristics that need to be 
            available for entity, as the intersection of calibrate_cols and firm_characteristics.
        save_TripleSort : boolean, Default=True
            If True, the triple_sorted dataframe is saved in save_folder.

        Attributes(Returns):
        ---------------------
        triple_sorted : DataFrame
            DataFrame in which entities/stocks have been triple sorted (conditionally or unconditionally).
        
        """
        
        
        # First firm characteristic
        # --------------------------
        self.firm_characteristic = firm_characteristics[0] 
        self.lagged_periods = lagged_periods[0]
        # Time arrow
        if self.lagged_periods > 0:
            self.Tarrow = 'lag'
        elif self.lagged_periods < 0:
            self.Tarrow = 'for'
        else:
            self.Tarrow_2 = ''
        # Portfolio name    
        if np.abs(self.lagged_periods)> 0 :
            self.portfolio = '%s_%s%d_portfolio' % (self.firm_characteristic, self.Tarrow, np.abs(self.lagged_periods) )
        else:
            self.portfolio = '%s_portfolio' % (self.firm_characteristic)
        # Portfolio quantiles for sorting   
        self.n_portfolios = n_portfolios[0] 
        # Number of portfolios if n_portfolios is not an integer
        if not isinstance(self.n_portfolios, int):
            self.num_portfolios = len(self.n_portfolios)
        else:
            self.num_portfolios = self.n_portfolios 
        # Filter quantiles 
        self.quantile_filter = quantile_filters[0]
        
        # Second firm characteristic
        # --------------------------
        self.firm_characteristic_2 = firm_characteristics[1]
        self.lagged_periods_2 = lagged_periods[1] 
        # Time Arrow 2
        if self.lagged_periods_2 > 0:
            self.Tarrow_2 = 'lag'
        elif self.lagged_periods_2 < 0:
            self.Tarrow_2 = 'for'
        else:
            self.Tarrow_2 = ''
        # Portfolio Name 2
        if np.abs(self.lagged_periods_2)>0:
            self.portfolio_2 = '%s_%s%d_portfolio' % (self.firm_characteristic_2, self.Tarrow_2, np.abs(self.lagged_periods_2) )
        else:
            self.portfolio_2 = '%s_portfolio' % (self.firm_characteristic_2)
        # Portfolio quantiles 2 for sorting
        self.n_portfolios_2 = n_portfolios[1] 
        # Number of portfolios if n_portfolios_2 is not an integer
        if not isinstance(self.n_portfolios_2, int):
            self.num_portfolios_2 = len(self.n_portfolios_2)
        else:
            self.num_portfolios_2 = self.n_portfolios_2
        # Filter quantiles 2
        self.quantile_filter_2 = quantile_filters[1]
            
        # Third firm characteristic
        # --------------------------
        self.firm_characteristic_3 = firm_characteristics[2]
        self.lagged_periods_3 = lagged_periods[2] 
        # Time Arrow 3
        if self.lagged_periods_3 > 0:
            self.Tarrow_3 = 'lag'
        elif self.lagged_periods_2 < 0:
            self.Tarrow_3 = 'for'
        else:
            self.Tarrow_3 = ''
        # Portfolio Name 3
        if np.abs(self.lagged_periods_3)>0:
            self.portfolio_3 = '%s_%s%d_portfolio' % (self.firm_characteristic_3, self.Tarrow_3, np.abs(self.lagged_periods_3) )
        else:
            self.portfolio_3 = '%s_portfolio' % (self.firm_characteristic_3)
        # Portfolio quantiles 3 for sorting
        self.n_portfolios_3 = n_portfolios[2] 
        # Number of portfolios if n_portfolios_3 is not an integer
        if not isinstance(self.n_portfolios_3, int):
            self.num_portfolios_3 = len(self.n_portfolios_3)
        else:
            self.num_portfolios_3 = self.n_portfolios_3                        
        # Filter quantiles3
        self.quantile_filter_3 = quantile_filters[2]        
        
        self.calibrate_cols = calibrate_cols
      
        
        # Conditional or unconditional sorting.
        self.conditional = conditional if conditional is not None else False
        # Define the save_folder 
        folder_name = '%dx%dx%d_portfolios_SortedBy_%sand%sand%s' % (self.num_portfolios, self.num_portfolios_2, \
                                                               self.num_portfolios_3, self.firm_characteristic, \
                                                               self.firm_characteristic_2, self.firm_characteristic_3)
        self.save_folder = os.path.join(self.save_dir, folder_name)
        if folder_name not in os.listdir(self.save_dir):
            os.mkdir(self.save_folder)
        self.save_TripleSort = save_TripleSort 
        
        
        # Function that defines the double sort portfolio
        def join_func(a,b):
            return '_'.join([str(int(a)), str(int(b))])
        
        # Function that defines the triple sort portfolio from double and and single
        def join_func2(a,b):
            return '_'.join([str(a), str(int(b))])
        
        # Function that defines the triple sort portfolio from single portfolios
        def join_func3(a,b,c):
            return '_'.join([str(int(a)), str(int(b)), str(int(c))])
        
        
        # Lag the three characteristics by lagged_periods so we can apply the Sort() function 
        # without lagged periods. 
        # -----------------------------------------------------------------------------------
        
        # Name of the lagged firm characteristic columns
        if np.abs(self.lagged_periods) > 0:
            firm_char1 = '%s_%s%d' %  (self.firm_characteristic,   self.Tarrow,   self.lagged_periods)
        else:
            firm_char1 = self.firm_characteristic
            
        if np.abs(self.lagged_periods_2):
            firm_char2 = '%s_%s%d' %  (self.firm_characteristic_2, self.Tarrow_2, self.lagged_periods_2)
        else:
            firm_char2 = self.firm_characteristic_2
            
        if np.abs(self.lagged_periods_3):
            firm_char3 = '%s_%s%d' %  (self.firm_characteristic_3, self.Tarrow_3, self.lagged_periods_3)
        else:
            firm_char3 = self.firm_characteristic_3
            
        # As a list
        firm_chars = [firm_char1, firm_char2, firm_char3]
        # Create a copy of the original dataframe to be used for triple sorting (_ts stands for triple sorting)
        df_ts = self.df.copy()
        # Apply the lagged_periods operator for firm_characteristic
        df_ts[firm_char1] = df_ts.groupby(self.entity_id)[self.firm_characteristic].apply(lambda x: x.shift(self.lagged_periods))
        # Apply the lagged_periods_2 operator for firm_characteristic_2
        df_ts[firm_char2] = df_ts.groupby(self.entity_id)[self.firm_characteristic_2].apply(lambda x: x.shift(self.lagged_periods_2))
         # Apply the lagged_periods_3 operator for firm_characteristic_3
        df_ts[firm_char3] = df_ts.groupby(self.entity_id)[self.firm_characteristic_3].apply(lambda x: x.shift(self.lagged_periods_3))
        
        # Drop rows for null values in firm_chars
        # Dataframe to be used in triple sorting
        if self.calibrate_cols is not None:
            df_ts = df_ts.dropna(subset = firm_chars + self.calibrate_cols)
        else:
            df_ts = df_ts.dropna(subset = firm_chars)
        
        
        # Check for conditional sort or not
        if self.conditional:
            
            
            # Single sort on firm_char1 with lagged_periods = 0 on df_ts
            single_sorted = self.Sort(df_ts, firm_characteristic = firm_char1, lagged_periods = 0, \
                                      n_portfolios = self.n_portfolios, quantile_filter = self.quantile_filter,\
                                      save_sort = False)

            # Double sort on firm_char2 with lagged periods = 0 on single_sorted dataframe
            double_sorted = single_sorted.groupby(self.portfolio).apply(lambda x:  self.Sort(x, \
                            firm_characteristic = firm_char2, lagged_periods = 0, \
                            n_portfolios = self.n_portfolios_2, quantile_filter = self.quantile_filter_2, \
                            save_sort = False) )
            # Reset the index
            double_sorted.reset_index(drop=True, inplace = True)
            
            
            
            # Define the double sort portfolio column
            double_sorted['Double_sort_portfolio'] = double_sorted.apply(lambda x: join_func(x[self.portfolio], x[self.portfolio_2]), axis = 1)
            
            
            # Triple sort on firm_char3 with lagged periods = 0 on double_sorted dataframe
            triple_sorted = double_sorted.groupby('Double_sort_portfolio').apply(lambda x:  self.Sort(x, \
                            firm_characteristic = firm_char3, lagged_periods = 0, \
                            n_portfolios = self.n_portfolios_3, quantile_filter = self.quantile_filter_3, \
                            save_sort = False) )
            # Reset the index
            triple_sorted.reset_index(drop = True, inplace = True)
            

            # Define the triple sort portfolio column
            triple_sorted['Triple_sort_portfolio'] = triple_sorted.apply(lambda x: join_func2(x['Double_sort_portfolio'], x[self.portfolio_3]), axis = 1)
                             

            # Define triple_sorted and sort again by entity and time.
            self.triple_sorted = triple_sorted.sort_values(by = [self.entity_id, self.time_id])
            
            if self.save_TripleSort:
                filename = 'conditional_triple_sort_on_%s_and_%s_and_%s.csv' % (self.firm_characteristic, \
                                                                                self.firm_characteristic_2,\
                                                                                self.firm_characteristic_3)
                self.triple_sorted.to_csv(os.path.join(self.save_folder, filename), index = False)

        else:
            # Apply self.Sort() three times in a row (nested functions)
            triple_sorted = self.Sort( self.Sort( self.Sort(df_ts, firm_char1, 0, self.n_portfolios, save_sort = False), firm_char2, \
                                       0, self.n_portfolios_2, save_sort = False), firm_char3, 0, self.n_portfolios_3, \
                                       save_sort = False)
   
                
            # Define triple sort portfolio columns
            triple_sorted['Triple_sort_portfolio'] = triple_sorted.apply(lambda x: join_func3(x[self.portfolio], x[self.portfolio_2], x[self.portfolio_3]), axis = 1)
              
            # Define triple_sorted and sort again by entity and time.
            self.triple_sorted = triple_sorted.sort_values(by = [self.entity_id, self.time_id]).reset_index(drop = True)
                        
                    
            if self.save_TripleSort:
                filename = 'unconditional_double_sort_on_%sand%sand%s.csv' % (self.firm_characteristic, \
                                                                              self.firm_characteristic_2,\
                                                                              self.firm_characteristic_3)
                self.triple_sorted.to_csv(os.path.join(self.save_folder, filename), index = False)
                
        
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #      SingleSortAgg         #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    
    # Single Sort entities on a characteristic and aggregate them on agg_var.
    def SingleSortAgg(self, firm_characteristic = 'CAP', lagged_periods = 1, n_portfolios = 10, \
                      quantile_filter = None, calibrate = True, add_cal_cols = [], \
                      save_SingleSortAgg = True, port_legends = None, agg_var = 'R2', \
                      agg_func = 'mean', agg_func_name = 'mean', \
                      label_names = ['Small', 'Large'], dpi_res = 500, num_port_stocks = True,
                      drop_first_dates = 0):
        """
        Function that sorts entities based on the firm_characteristic of the previous lagged_periods periods
        and aggregate them based on agg_var. Plots of the aggregated values are also returned.
    
        Attributes:
        -----------
        calibrate : boolean, Default=True
            If True, only stocks with avalaible data on agg_var and firm_characteristic
            by which they are sorted are used in the sorting and aggregation procedure. 
            For example, there are some stocks for which the lagged_periods value of 'CAP' (firm_characteristic)  
            is available but 'Total Assets' (agg_var)  may not be. 
        save_SingleSortAgg : boolean, Default=True
            If True, the results are saved in the save_folder directory.
        port_legends : list, Default=['Low', 'High']
            The list contains the names of the first (bottom quantile) 
            and last (top quantile) portfolio. 
        agg_var: string, Default='R2'
            The variable on which data will be aggregated within the sorted portfolios.
            The default value of 'R2' was data-specific and one is free to change it to 
            whatever value makes sense for him/her. 
        agg_func : function, Default= lambda x: x.mean()
            The function by which agg_var should be aggregated along the portfolio dimension.
        agg_func_name: string, Default = 'mean'
            Name of the aggregated function used in the name of aggregated dataframe and figure.
            If the agg_func is changed, the agg_func_name should be changed too to reflect 
            the different aggregate function.
        label_names : list, Default=['Year', agg_var']
            The first eleement of the list is the name of x-axis and the second the name of y-axis.
        dpi_res : integer, Default=600
            The resolution of the figures when saved as images in save_folder.
        num_port_stocks : boolean, Default=True
            If True, the number of stocks in the portfolos for each year are also plotted and returned. 
            The number of stocks might differ from the one expected by the sort: 
            1. Calibrate = False: 
                The agg_var might not be available for all stocks in the portfolios. 
            2. Calibrate = True:
                The way entities/stocks are assigned to portfolios gives rise to 
                differences in the number of stocks in each portfolio. 
        add_cal_cols : list, Default = []
            A list of columns to be added in the aggregation procedure so that null values can be 
            purged from the data before aggregating on agg_var. Generally, add_cal_cols are used
            when the agg_func uses some other column of the data to aggregate agg_var. Notable examples 
            are weighted means instead of simple means; consider the case of market cap weighted portfolios.
        drop_first_dates : integer, Default=0
            The number of years/dates to be dropped at the start of the full sample. 

        Attributes(Returns):
        --------------------
        sorted_agg : DataFrame
            The assign portfolio column is created and populated.
        data_agg : DataFrame
            Dataframe with index = Date, columns = Portfolios and values = aggregated data.
        fig_agg : figure object
            Figure object for plotting the aggregated data.
        ax_agg : axes object
            Axes object for plotting the aggregated data.
        num_stocks : dataframe
            Dataframe contatining the number of stocks in each portfolio for each time period.
        fig_num : figure object
            Figure object for plotting the number of stocks in each portfolio.
        ax_num : axes object
            Axes object for plotting the number of stocks in each portfolio.
            
                           
        """
        
        # Instance Variables 
        self.firm_characteristic 
        self.lagged_periods = lagged_periods 
        # Time Arrow
        if self.lagged_periods > 0:
            self.Tarrow = 'lag'
        elif self.lagged_periods < 0:
            self.Tarrow = 'for'
        else:
            self.Tarrow = ''
        # Portfolio name    
        if np.abs(self.lagged_periods)> 0 :
            self.portfolio = '%s_%s%d_portfolio' % (self.firm_characteristic, self.Tarrow, np.abs(self.lagged_periods) )
        else:
            self.portfolio = '%s_portfolio' % (self.firm_characteristic)
        # Portfolio quantiles for sorting
        self.n_portfolios = n_portfolios 
        # Number of portfolios if n_portfolios is not an integer
        if not isinstance(n_portfolios, int):
            self.num_portfolios = len(self.n_portfolios)
        else:
            self.num_portfolios = self.n_portfolios 
        # There is no default value of quantile_filter
        self.quantile_filter = quantile_filter 
        # Define the save_folder 
        folder_name = '%d_portfolios_SortedBy_%s' % (self.num_portfolios, self.firm_characteristic)
        self.save_folder = os.path.join(self.save_dir, folder_name)
        
        self.add_cal_cols = add_cal_cols 
        self.calibrate = calibrate 
        self.save_SingleSortAgg = save_SingleSortAgg 
        self.port_legends = port_legends 
        self.agg_var = agg_var 
        self.agg_func = agg_func 
        self.agg_func_name = agg_func_name 
        self.label_names = label_names if label_names is not None else [self.time_id, self.agg_var]
        self.dpi_res = dpi_res if dpi_res is not None else 500
        self.num_port_stocks = num_port_stocks 
        self.drop_first_dates = drop_first_dates 
       
       
        # Define DF dataframe 
        DF = self.df.copy(deep = True)

        
        # Calibrate data for both sorting and aggregating.
        if self.calibrate:
            
            # Name of the new column
            if self.lagged_periods > 0:
                cal_col = self.firm_characteristic + '_lag%d' % self.lagged_periods
            elif self.lagged_periods < 0:
                cal_col = self.firm_characteristic + '_forw%d' % self.lagged_periods
            else:
                cal_col = self.firm_characteristic
                
           
            

            
            # Align the lagged firm_characteristic and the aggregated variable
            DF[cal_col] = DF.groupby(by = self.entity_id)[self.firm_characteristic].shift(periods = self.lagged_periods)
            

           

            # Delete observations where both the lagged firm characteristic and aggragated var are missing.
            # ---------------------------------------------------------------------------------------------
    
            # This operation ensures that only entities with a non-missing agg_var will be used in the 
            # sorting procedure. That way in a year, the number of stocks in a given portfolio will be the same
            # as only observations with a non-missing value of firm characteristic (lagged) and agg_var will be used in the 
            # sorting. Any deviation in the number of entities in the portfolios in a year is the result of the definition of 
            # the quantiles and in turn of the definition of the rank of the portfolio. 
            DF.dropna(subset = [cal_col, self.agg_var] + self.add_cal_cols, how = 'any', inplace = True)
            DF.reset_index(drop = True, inplace = True)

            

            
            port_char = self.Sort(df = DF, firm_characteristic = cal_col, lagged_periods = 0, \
                                n_portfolios = self.n_portfolios, quantile_fitler = self.quantile_filter, \
                                 prefix_name = self.agg_var + '_agg')
                
            
                
        else:
            
            DF.dropna(subset = [self.agg_var] + self.add_cal_cols, how = 'any', inplace = True)
            # Call SortPortfolios function on firm_characteristic and lagged periods = lagged periods
            port_char = self.Sort(df = DF, firm_characteristic = self.firm_characteristic, \
                                lagged_periods = self.lagged_periods, n_portfolios = self.n_portfolios, \
                                quantile_fitler = self.quantile_filter, prefix_name = self.agg_var + '_agg')
            
          
    
        
        # Name of portfolio column
        if self.lagged_periods > 0:
            port_col =  '%s_lag%d_portfolio' % (self.firm_characteristic, self.lagged_periods)
        elif self.lagged_periods < 0:
            port_col =  '%s_for%d_portfolio' % (self.firm_characteristic, self.lagged_periods)
        else:
            # In case sorting should be done on contemporaneous values
            port_col =  '%s_portfolio' % self.firm_characteristic
            
        # Define the name of the aggregated data files (dataframe and figure)
        save_name = '%s_%s_of_%d_portfolios_basedOn_%s' % (self.agg_func_name, self.agg_var, \
                                                           self.num_portfolios, self.firm_characteristic)
       
        
        # Drop the first dates as specified by drop_first_dates
        dates_to_keep = port_char[self.time_id].value_counts().sort_index().index[self.drop_first_dates:] 
        # Keep the rest dates
        port_char = port_char[port_char[self.time_id].isin(dates_to_keep)]
        
        self.sorted_agg = port_char
            
        # Aggregate on 'agg_var' along portfolios based on firm_characteristic
        port_char_agg = port_char.groupby(by = [port_col, self.time_id] ).agg( { self.agg_var : self.agg_func } ).unstack(level=0)
        # Rename the columns
        port_char_agg.columns = [x[1] for x in port_char_agg.columns]
        
        self.data_agg = port_char_agg
        
        # Generate legend for portfolios
        legend_ls = [self.port_legends[0]] + list(range(2,self.num_portfolios))  + [self.port_legends[1]]
            
        # Plot the aggregated value of agg_var of the portfolios based on firm_characteristic
        fig_agg = plt.figure()
        ax_agg = fig_agg.add_subplot(1, 1, 1)
        port_char_agg.plot(linewidth = 1, ax = ax_agg)
        ax_agg.set_xlabel(self.label_names[0])
        ax_agg.set_ylabel(self.label_names[1])
        ax_agg.legend(legend_ls, loc ='upper left', fontsize = 'x-small')
        # Save figure
        if self.save_SingleSortAgg:
            plt_save_path = os.path.join(self.save_folder, save_name+'.png')
            fig_agg.savefig(plt_save_path, bbox_inches='tight', dpi = self.dpi_res)
        self.fig_agg = fig_agg
        self.ax_agg = ax_agg
        
    
        # Calculate difference of the aggregated variable between first and last portfolio denoted 
        # by the column named port_legends[1] - port_legends[0]. 
        diff_col_name = ' - '.join(self.port_legends[::-1])
        port_char_agg[diff_col_name] = port_char_agg.iloc[:,-1] - port_char_agg.iloc[:,0]
        # Save aggregated dataframe
        if self.save_SingleSortAgg: 
            # save dataframe
            csv_save_path = os.path.join(self.save_folder, save_name+'.csv')
            port_char_agg.to_csv(csv_save_path)
            
        if self.num_port_stocks:
            fig_num = plt.figure()
            ax_num = fig_num.add_subplot(1, 1, 1)
            self.num_stocks = port_char.groupby([port_col, self.time_id])[self.agg_var].count().unstack(level = 0)
            self.num_stocks.plot(linewidth = 1, ax = ax_num)
            ax_num.set_xlabel('Year')
            ax_num.set_ylabel('Number of Stocks')
            ax_num.legend(legend_ls, loc ='upper left', fontsize = 'x-small')
            fig_num.savefig(os.path.join(self.save_folder, save_name + '_number.png'), bbox_inches='tight', dpi = self.dpi_res)
            
            self.fig_num = fig_num
            self.ax_num = ax_num
    
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #      FFPortfolios          #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
    # Function that generates the weighted average returns of portfolios of entities that have been sorted 
    # by their characteristics. 
    def FFPortfolios(self, ret_data, ret_time_id, FFcharacteristics, FFlagged_periods, \
                 FFn_portfolios, FFquantile_filters, FFdir = None, FFconditional = False, weight_col = None, \
                 return_col = 'RET', FFsave = True):
        """
        
        Parameters
        ----------
        ret_data : Dataframe
            Dataframe where returns for entities are stored in a panel format.
        ret_time_id : str
            Time identifier as found in ret_data. ret_time_id dicates the frequency for which the portfolio
            returns are calculated.
        FFcharacteristics : list
            A list of up to three characteristics for which entities will be sorted.
        FFlagged_periods : list
            A list of the number of lagged periods for the characteristics to be sorted.
            The length of characteristics and lagged_periods must match.
        FFn_portfolios : list 
            N_portfolios is a list of n_portfolios.
            If n_portfolios then stocks will be sorted in N_portfolios equal portfolios.
            If n_portfolios is an array then this array represents the quantiles. 
        FFquantile_filters : list
            It is a list of lists. Each element corresponds to filtering entities for the 
            ranking of portfolios into each firm characteristic. The lenght of the list must 
            be the same as that of firm_characteristics.
        FFdir : directory
            Saving directory.
        FFconditional: boolean, Default=False
            If True, all sorts are conditional. If False, all sorts are unonconditional and 
            independent of each other. 
        weight_col : str, Default=None
            The column used for weighting the returns in a portfolio. If weight_col is None,
            the portfolios are equal-weighted.
        return_col : str, Default='RET'
            The column of ret_data that corresponds to returns. Default value is 'RET' which is the 
            name of return for CRSP data.
        FFsave : boolean, Default=True
            If True, save results to FFdir.
            
        Attributes(returns)
        -------------------
        FFportfolios : DataFrame
            Dataframe with columns = portfolios, index = ret_time_id, values = returns
        FFnum_stocks : DataFrame
            Dataframe with columns = portfolios, index = ret_time_id, values = number of stocks in each portfolio
                
        
        """
        
        # Instance variables
        self.ret_data = ret_data
        self.ret_time_id = ret_time_id
        self.FFcharacteristics = FFcharacteristics
        self.FFlagged_periods = FFlagged_periods
        self.FFn_portfolios = FFn_portfolios
        self.FFquantile_filters = FFquantile_filters
        self.FFdir = FFdir if FFdir is not None else self.save_dir
        self.FFconditional = FFconditional
        self.weight_col = weight_col
        self.return_col = return_col
        self.FFsave = FFsave
        
        
        # Create a copy of the original FirmCharacteristics dataframe
        firmchars = self.df.copy()
        
        # Create a new PortSort class with df = firmchars
        FFclass = PortSort(df = firmchars, entity_id = self.entity_id, time_id = self.time_id, \
                           prefix_name = self.prefix_name, save_dir = self.save_dir)
    
        # weight_col as the calibration column
        if self.weight_col is not None:   
            FFcalibrate_col = [self.weight_col]
        else:
            FFcalibrate_col = None


        # -----------------------------------
        #  SORT -- SINGLE or DOUBLE or TRIPLE
        # -----------------------------------
        
        
        # One characteristic --> Single Sort
        # ----------------------------------
        if len(FFcharacteristics) == 1:
            
             
            # Single sort
            FFclass.SingleSort(firm_characteristic = self.FFcharacteristics[0], n_portfolios = self.FFn_portfolios[0], \
                            lagged_periods = self.FFlagged_periods[0],  quantile_filter = self.FFquantile_filters[0],\
                            calibrate_cols = FFcalibrate_col, save_SingleSort = self.FFsave) 
            
            
            # Isolate only the essential columns for portfolio assignment
            port_name = FFclass.portfolio
            # Include or not weighting column
            if self.weight_col is not None:
                ports = FFclass.single_sorted[[self.time_id, self.entity_id, self.weight_col, port_name]].copy()    
            else:
                ports = FFclass.single_sorted[[self.time_id, self.entity_id, port_name]].copy()                
            
            
            # Define save names
            save_str =  '%d_portfolios_sortedBy_%s.csv' % (FFclass.num_portfolios, self.FFcharacteristics[0])
            save_ret = 'RET_' + save_str
            save_num = 'NUM_STOCKS_' + save_str
            
        # Two characteristic --> Double Sort
        # -----------------------------------
        if len(FFcharacteristics) == 2:
                   
            
            # Double sort
            FFclass.DoubleSort(firm_characteristics = self.FFcharacteristics, lagged_periods = self.FFlagged_periods,\
                            n_portfolios = self.FFn_portfolios, quantile_filters = self.FFquantile_filters, \
                            conditional = self.FFconditional, calibrate_cols = FFcalibrate_col, save_DoubleSort = self.FFsave)   
            
            # Isolate only the essential columns for portfolio assignment
            port_name = FFclass.double_sorted.columns[-1]
            # Include or not weighting column
            if self.weight_col is not None:
                ports = FFclass.double_sorted[[self.time_id, self.entity_id, self.weight_col, port_name]].copy()
            else:
                ports = FFclass.double_sorted[[self.time_id, self.entity_id, port_name]].copy()
            
            # Define save names
            save_str =  '%dx%d_portfolios_sortedBy_%sand%s.csv' % (FFclass.num_portfolios, FFclass.num_portfolios_2, \
                                                                     self.FFcharacteristics[0], self.FFcharacteristics[1])
            save_ret = 'RET_' + save_str
            save_num = 'NUM_STOCKS_' + save_str
            
        
        # Three characteristics --> Triple Sort
        # --------------------------------------
        if len(self.FFcharacteristics) == 3:
            
            
            # Triple sort
            FFclass.TripleSort(firm_characteristics = self.FFcharacteristics, lagged_periods = self.FFlagged_periods, \
                            n_portfolios = self.FFn_portfolios, quantile_filters = self.FFquantile_filters, \
                            conditional = self.FFconditional, calibrate_cols = FFcalibrate_col, save_TripleSort = self.FFsave)
    
            # Isolate only the essential columns for portfolio assignment
            port_name = FFclass.triple_sorted.columns[-1]
            # Include or not weighting column
            if self.weight_col is not None:
                ports = FFclass.triple_sorted[[self.time_id, self.entity_id, self.weight_col, port_name]].copy()  
            else:
                ports = FFclass.triple_sorted[[self.time_id, self.entity_id, port_name]].copy()   
    
            
            # Define save names
            save_str =  '%dx%dx%d_portfolios_sortedBy_%sand%sand%s.csv' % ( FFclass.num_portfolios, \
                                                                        FFclass.num_portfolios_2, \
                                                                        FFclass.num_portfolios_3, \
                                                                        self.FFcharacteristics[0], \
                                                                        self.FFcharacteristics[1],\
                                                                        self.FFcharacteristics[2])
            save_ret = 'RET_' + save_str
            save_num = 'NUM_STOCKS_' + save_str
                
    
        
        
        # Number of stocks in a portfolio
        # -------------------------------
        num_stocks = ports.groupby(by = [port_name, self.time_id] )[port_name].count().unstack(level=0)
        
        
        # ----------------------------------
        #  ASSIGN PORTFOLIOS TO RETURN DATA 
        # ----------------------------------
        
        # The inner merging is taking care of stocks that should be excluded from the formation of the portfolios
        ret_ports = pd.merge(self.ret_data, ports, how = 'inner', on = [self.time_id, self.entity_id], suffixes = ('', '_2') )
        
        # Equal weighted portfolios or not
        if self.weight_col is None:
            char_ports = ret_ports.groupby(by = [port_name, self.ret_time_id] )[self.return_col].mean().unstack(level=0)
        else:
            char_ports = ret_ports.groupby(by = [port_name, self.ret_time_id] ).agg( { self.return_col : lambda x: WeightedMean(x, df = ret_ports, weights = self.weight_col) } ).unstack(level=0)
            # Rename the columns by keeping only the second element of their names
            char_ports.columns = [x[1] for x in char_ports.columns]
        
        #-------------
        # SAVE RESULTS
        # ------------
                
        char_ports.to_csv(os.path.join(FFclass.save_folder, save_ret ))
        num_stocks.to_csv(os.path.join(FFclass.save_folder, save_num ))
        
        # Attributes
        self.FFportfolios = char_ports
        self.FFnum_stocks = num_stocks
        # Return the class
        self.FFclass = FFclass

        
    

      
        




        
        
        


