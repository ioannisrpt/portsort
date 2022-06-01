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
        sort()         
        single_sort()     
        double_sort()       
        triple_sort()
        ff_portfolios()
        
        Version 0.4.0 - 1 Jun 2022
        
        i. Code is re-written according to PEP 8 guidelines.
        ii. The weights for averaging returns are calculated and 
        applied correctly at the start of the rebalancing period.
        iii. ff_portfolios() is augmented with the capability of creating
        placeholder rows for securities that are delisted but are used
        for sorting. If we do not take delistings into account, we introduce
        a look-ahead bias in our portfolios.
        
        Version 0.3.1 - 19 Feb 2022
        
        i. All processes are vectorized. All unecessary apply lambda operations
        have been replaced. The sort function and the PortSort sorting methods 
        are now 10 times faster. FFPortfolios() method is now 3-5 times faster.
        
        
        Version 0.2.7 - 14 Feb 2022

        
        i. Conditional sorts can now be in the form of TF=[True, False], TT, 
        FT, FF in the triple sort procedure.          
        ii. Fixed a potential bug in calibrating the return_col in 
        FFportfolios().
        iii. WeightedMean function ignores nan values in both the aggregated 
        column x and the weight column weights. 
"""



import os
import pandas as pd
import numpy as np




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         WEIGHTED MEAN IN A DATAFRAME                #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Weighted mean ignoring nan values 
def weighted_mean(x, df, weights):
    """
    Define the weighted mean function
    """
    # Mask both the values and the associated weights
    ma_x = np.ma.MaskedArray(x, mask=np.isnan(x))
    w = df.loc[x.index, weights]
    ma_w = np.ma.MaskedArray(w, mask=np.isnan(w))
    return np.average(ma_x, weights=ma_w)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         PREPARE DATA FOR SORTING PORTFOLIOS (GENERAL CHARACTERISTIC)     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def prepare_for_sorting(df, 
                       entity_id, 
                       time_id, 
                       firm_characteristic, 
                       lagged_periods, 
                       n_portfolios, 
                       quantile_filter = None):
       
    """
    Prepare DataFrame for sorting across one column.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe that contains information in a panel format.
    entity_id : str
        This is the column that corresponds to the entities.
    time_id : str
        time_id denotes the time dimension the panel dataset. 
    firm_characteristic : str
        Sort portfolios based on firm characteristic. It must exist in the 
        column of df.
    lagged_periods : int
        The value of the firm characteristic used for sorting would be the 
        value lagged_periods before.
    n_portfolios : int or array
        If n_portfolios is integer, then quantiles will be calculated as 
        1/n_portfolios so that n_portfolios will be constructed.
        If n_portfolios is an array, then quantiles will be calculated 
        according to the array.
   quantile_filter : list, optional
        quantile_filter is a list = [column_name, value] with the first element 
        being the name of the column and the second being the value for which
        the quantiles should be calculated only. For example, if 
        quantile_filter = ['EXCHCD', 1], then only NYSE stocks will be used for
        the estimation of quantiles. If it is None, all entities will be used.
       
    Returns:
    ---------
    df: DataFrame
        The returned df contains the new column, 'firm_characteristic_end', 
        that is the adjusted characteristic used for sorting. Rows that have 
        null values in 'firm_characteristic_end' are dropped.
        
    trait_q: Series
        Series with index = (time_id, probability), value = quantiles and 
        name = firm_characteristic_end
                       
    """
    
    # For simplicity, we denote the firm characteristic with a different name 
    trait = firm_characteristic
   

    # Define the lagged or lead firm characteristic 
    # ---------------------------------------------
    
    # Name of the new column
    if lagged_periods > 0:
        trait_end = trait + '_lag%d' % lagged_periods
    elif lagged_periods < 0:
        trait_end = trait + '_for%d' % np.abs(lagged_periods)
    else:
        trait_end = trait
    # Adjust the characteristic for sorting by entity_id
    df[trait_end] = (
                    df.groupby(by=entity_id)[trait]
                     .shift(periods=lagged_periods)
                     )
    # Drop rows with null values in 'trait_end'. This is the only criterion 
    # by which null values are dropped. Only 'trait_end' matters for sorting.
    df = df.dropna(subset=[trait_end])
    
    
    
    # Quantiles for each year based on trait_end and n_portfolios 
    # -----------------------------------------------------------

    # Check if n_portfolios is integer 
    if isinstance(n_portfolios, int):
        q_range = np.arange(start=0, stop=1, step=1/n_portfolios)
    # Else it is array and it is used as it is
    else:
        q_range = n_portfolios
        
    # Definition of quantiles 
    # -----------------------
    
    if quantile_filter is None:
        trait_q = (
                  df.groupby(by=[time_id])[trait_end]
                  .quantile(q_range, interpolation='linear')
                  )
    else:
        # Apply filter for estimation of quantiles
        fil_col = quantile_filter[0]
        fil_value = quantile_filter[1]
        trait_q = (
                  df[df[fil_col]==fil_value]
                  .groupby(by=time_id)[trait_end]
                  .quantile(q_range, interpolation='linear')
                  )

    return df, trait_q


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           ASSIGN QUANTILE TO AN ENTITY           #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def assign_quantiles(data, quantiles):
    
    """
    Assign quantile ranking to an entity.
    
    Parameters
    ----------
    data : Series 
        A series that contains the values of the firm characteristic 
        to be sorted (for a given period time_id).
    quantiles : Series
        A Series that contains the quantiles that are assigned to each entity. 
        Index = probability, value = quantile, name = sorted characteristics


    Returns
    -------
    data_q : Series
        Series of same length as data that contains the quantile portfolio 
        for each entity based on quantiles.

    """
    
    # Number of portfolios/total quantiles
    n_portfolios = len(quantiles)
    
    # Create a new series just like data. Fill it with the maximum portfolio 
    # value
    data_q = pd.Series(data=n_portfolios, index=data.index, name=data.name)

    
    # Iterate through the quantile values and assign them to entities.
    # The intervals (q_{i-1}, q_{i}] for i = 1 to n_portfolios are defined.
    for interval in range(1, n_portfolios): 
        # The left side of the interval becomes equality so that the 
        # minimum value can be included in that first quantile.
        if interval == 1:
            mask1 = quantiles.iloc[interval - 1] <= data    
        else:
           mask1 = quantiles.iloc[interval - 1] < data   
        mask2 = data <= quantiles.iloc[interval]
        data_q.loc[ mask1 & mask2 ] = int(interval)

    return data_q


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#        ASSIGN THE QUANTILE PORTFOLIO TO AN ENTITY          #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Function that assigns the quantile of the portfolio for a firm characteristic
def assign_portfolio(df, df_q, time_id):
    
    """
    Assign portfolio ranking to an entity.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe that contains all relevant data in a panel format.
    df_q : DataFrame
        A Series that contains the quantiles that are assigned to each entity. 
        Index = probability, value = quantile, name = sorted characteristics
    time_id : str
        Time dimension of the panel dataset. 
    save_dir : dir 
        Directory where the results are saved.

    Returns
    -------
    df_c: DataFrame
        Same dataframe as df that contains the new portfolio column with
        values 1 to n_portfolios for each entity based on the quantiles of 
        the sorting characteristic per period.

    """
    
    # Retrieve Dates/Years; df_q is a multiindex series by definition
    dates = df_q.index.unique(level=0)
    
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
        df_c.loc[date_idx, col_name] = (
            assign_quantiles(data = df_c.loc[date_idx, firm_characteristic],
                            quantiles = quantiles)
            )
        
    
    return df_c 



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#               CLASS     PortSort                         #         
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




class PortSort:
    
    # Class Variables
    
    # Instance Variables
    def __init__(self, 
                 df, 
                 entity_id = None, 
                 time_id = None,  
                 prefix_name = None,  
                 save_dir = None):

        
        self.entity_id = entity_id if entity_id is not None else 'PERMNO'
        self.time_id = time_id if time_id is not None else 'Date'
        # Create a copy of the input dataframe df. Drop duplicate values and 
        # sort by ['entity_id', 'time_id'] to ensure that lagging or forwarding
        # a firm_characteristic results in the correct sorting characteristic
        self.df = (df
                   .drop_duplicates(subset=[self.entity_id, self.time_id], 
                                    ignore_index=True)
                   .sort_values(by = [self.entity_id, self.time_id])
                   .copy(deep = True)
                   )
        self.prefix_name = prefix_name if prefix_name is not None else ''
        self.save_dir = save_dir if save_dir is not None else os.getcwd()



               

    # ~~~~~~~~~~~~~~~~~~~
    #        Sort       #
    # ~~~~~~~~~~~~~~~~~~~
        
    # Sort entities by characteristic. 
    def sort(self, 
             df = None, 
             firm_characteristic = 'CAP', 
             lagged_periods = 1,
             n_portfolios = 10,
             quantile_filter = None, 
             prefix_name = None, 
             save_sort = False):
        
        """
        Sort entities into n_portfolios based on firm_characteristic.
                
    
        Parameters
        ----------
        df : DataFrame, optional
            Input dataframe in a panel data format. 
            If None, then self.df is used.
        firm_characteristic : str, default 'CAP'
            Portfolios are sorted based on firm_characteristic. 
            It must exist on the column of df.
        n_portfolios : int or numpy.array, default 10
            Number of portfolios that are going to be constructed. 
            Decile portfolios are constructed by default.
        lagged_periods : int, default 1
            The value of the firm characteristic used would be the value 
            lagged_periods before. 
            Default is 1 so the characteristic one period before will be used 
            to sort entities into portfolios.
        quantile_filter : list, optional
            quantile_filter is a list = [column_name, value] with the first 
            element being the name of the column and the second being the value
            for which the quantiles should be calculated only. For example, if 
            quantile_filter = ['EXCHCD', 1], then only NYSE stocks will be used 
            for the estimation of quantiles. If it is None, all entities will 
            be used.       
        prefix_name : str, default ''
            The prefix of the name of the new dataframe as saved in save_dir. 
        save_sort : bool, default False
            If True then returns are saved in save_dir.

       
        Returns
        -------
        df_sorted : DataFrame
            Dataframe where entities have been assigned to a portfolio based 
            on firm_characteristic.              
        """
        
        # Variables only inside function
        df = df.copy(deep=True) if df is not None else self.df.copy(deep=True)       
        # Count the number of portfolios 
        if not isinstance(n_portfolios, int):
            num_portfolios = len(n_portfolios)
        else:
            num_portfolios = n_portfolios    
            
        # Define the save_folder
        folder_name = (
                      '%d_portfolios_SortedBy_%s' % 
                       (num_portfolios, firm_characteristic)
                       )
        save_folder = os.path.join(self.save_dir, folder_name)

        
        
        # Prepare the data for sorting and define quantiles based on 
        # firm characteristic
        df_trait, trait_q = prepare_for_sorting(df, 
                                               self.entity_id, 
                                               self.time_id, 
                                               firm_characteristic, 
                                               lagged_periods, 
                                               n_portfolios, 
                                               quantile_filter) 
        
        # Assign portfolio to each stock in our sample
        df_sorted = assign_portfolio(df_trait, trait_q, time_id=self.time_id)
        
        if save_sort:
            
            if folder_name not in os.listdir(self.save_dir):
                os.mkdir(save_folder)
            # Save portfolios dataframe
            fnm_ports = (
                        '%s_of_%d_portfolios_basedOn_%s.csv' % 
                        (prefix_name, num_portfolios, firm_characteristic) 
                        )
            df_sorted_path = os.path.join(save_folder, fnm_ports)
            df_sorted.to_csv(df_sorted_path, index=False)
            # Save quantiles dataframe
            fnm_quant = (
                        '%s_quantiles_of_%d_portfolios.csv' % 
                        (firm_characteristic, num_portfolios)
                        )
            trait_q_path = os.path.join(save_folder, fnm_quant)
            trait_q.to_csv(trait_q_path)
        
       
        return df_sorted
    
    

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    #      single_sort         #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Single Sorting 
    def single_sort(self, 
                   firm_characteristic, 
                   lagged_periods = None,  
                   n_portfolios = None, 
                   quantile_filter = None, 
                   calibrate_cols = None, 
                   save_SingleSort = False):
        """
        Sort entities based on one characteristic.
                
        Parameters:
        -----------
        firm_characteristic : str
            Portfolios are sorted based on a single firm_characteristic. 
        lagged_periods : int, default 1
            The value of the firm characteristic used would be the value 
            lagged_periods before. 
            Default is 1 so the characteristic one period before would be 
            used to sort entities into portfolios.
        n_portfolios : int or numpy.array, default 10
            Number of portfolios that are going to be constructed. 
            Decile portfolios are constructed by default.
        quantile_filter : list, optinal
            quantile_filter is a list = [column_name, value] with the first 
            element being the name of the column and the second being the value
            for which the quantiles should be calculated only. For example, if 
            quantile_filter = ['EXCHCD', 1], then only NYSE stocks will be used 
            for the estimation of quantiles. If it is None, all entities will 
            be used. 
        calibrate_cols : list, optional
            Only entities that have non-null values of calibrate_cols, 
            are sorted based on their firm_characteristic. We restrict the set
            of characteristics that need to be available for entity,
            as the union of calibrate_cols and firm_characteristics.
        save_SingleSort : bool, default True
            If True, the double_sorted dataframe is saved in save_dir.

        Returns:
        --------
        single_sorted : DataFrame
            DataFrame in which entities have been single sorted.       
        """        
        

        self.firm_characteristic = firm_characteristic
        self.lagged_periods = (
            lagged_periods if lagged_periods is not None else 1
            )
        # Time Arrow
        if self.lagged_periods > 0:
            self.Tarrow = 'lag'
        elif self.lagged_periods < 0:
            self.Tarrow = 'for'
        else:
            self.Tarrow = ''
        # Portfolio name    
        if np.abs(self.lagged_periods)> 0 :
            self.portfolio = (
                '%s_%s%d_portfolio' % 
                (self.firm_characteristic,
                 self.Tarrow, 
                 np.abs(self.lagged_periods))
                )
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
        folder_name = (
            '%d_portfolios_SortedBy_%s' % 
            (self.num_portfolios, self.firm_characteristic)
            )
        self.save_folder = os.path.join(self.save_dir, folder_name)
        if folder_name not in os.listdir(self.save_dir):
            os.mkdir(self.save_folder)


           
        # Lag the single characteristic by lagged_periods so we can apply 
        # the Sort() function without lagged periods. 
        # Account for calibrate_cols
        # -------------------------------------------------------------------
        
        # Name of the lagged firm characteristic 
        if np.abs(self.lagged_periods) > 0:
            firm_char1 = (
                '%s_%s%d' %  
                (self.firm_characteristic, self.Tarrow, self.lagged_periods)
                )
        else:
            firm_char1 = self.firm_characteristic
        # As a list
        firm_chars = [firm_char1]
        # Create a copy of the original dataframe to be used 
        # for double sorting (_ss stands for single sorting) 
        df_ss = self.df.copy()
        # Apply the lagged_periods operator for firm_characteristic
        df_ss[firm_char1] = (df_ss
                             .groupby(self.entity_id)[self.firm_characteristic]
                             .shift(self.lagged_periods)
                             )      
        
        # Drop rows for null values in firm_chars and calibrate_cols
        # Dataframe to be used in single sorting
        if self.calibrate_cols is not None:
            df_ss = df_ss.dropna(subset=firm_chars + self.calibrate_cols)
        else:
            df_ss = df_ss.dropna(subset=firm_chars)
    
    
        # ---------------------------
        # SORTING THE CHARACTERISTIC
        # ---------------------------

        self.single_sorted = (
                               self.sort(df = df_ss, 
                                       firm_characteristic = firm_char1, 
                                       lagged_periods = 0, 
                                       n_portfolios = self.n_portfolios, 
                                       quantile_filter = self.quantile_filter,
                                       prefix_name = self.prefix_name, 
                                       save_sort = save_SingleSort)
                              .reset_index(drop  = True)
                              )
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~
    #     double_sort      #
    # ~~~~~~~~~~~~~~~~~~~~~~
        
    # Double Sorting
    def double_sort(self, 
                   firm_characteristics, 
                   lagged_periods = None,
                   n_portfolios = None, 
                   quantile_filters = [None, None], 
                   conditional = False,
                   calibrate_cols = None, 
                   save_DoubleSort = False):
        
        """
        Sort entities based on two characteristics.
        
        Parameters:
        -----------
        firm_characteristics : list of str
            First element = first firm characteristic
            Second element = second firm characteristic
        lagged_periods : list of int 
            First element = lagged periods for first characteristic
            Second element = lagged periods for second characteristic
        n_portfolios : list of int or numpy.array
            First element = portfolios for sorting on first characteristic 
            Second element = portfolios for sorting on second characteristic 
        quantile_filters : list of list
            First element = quantile filter for first characteristic
            Second element = quantile filter for second characteristic            
        conditional : bool, default False
            If True, the second sort is conditional on the first. 
            If False, the sorts are indepedent. 
        calibrate_cols : list of str, optional
            Only entities that have non-null values of calibrate_cols, 
            are sorted based on their firm_characteristics. We restrict 
            the set of characteristics that need to be available for an entity, 
            as the union of calibrate_cols and firm_characteristics.
        save_DoubleSort : bool, default True
            If True, the double_sorted dataframe is saved.


        Returns:
        --------
        double_sorted : DataFrame
            DataFrame in which entities have been double sorted 
            conditionally or unconditionally.      
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
            self.portfolio = (
                '%s_%s%d_portfolio' % 
                (self.firm_characteristic, 
                 self.Tarrow, 
                 np.abs(self.lagged_periods))
                )
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
            self.portfolio_2 = (
                '%s_%s%d_portfolio' % 
                (self.firm_characteristic_2, 
                 self.Tarrow_2, 
                 np.abs(self.lagged_periods_2))
                )
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
        self.conditional = conditional 
        # Differentiate between conditional and unconditional sorts
        # 'D' stands for dependent sorts
        # 'I' stands for independent sorts
        if self.conditional:
            self.c1 = 'D'
        else:
            self.c1 = 'I'
        # Define the save_folder 
        folder_name = ( 
            '%dx%d_%s_portfolios_SortedBy_%sand%s' % 
            (self.num_portfolios, 
             self.num_portfolios_2, 
             self.c1, 
             self.firm_characteristic,
             self.firm_characteristic_2)
            )
        self.save_folder = os.path.join(self.save_dir, folder_name)
        if folder_name not in os.listdir(self.save_dir):
            os.mkdir(self.save_folder)
        # Save results
        self.save_DoubleSort = save_DoubleSort 
        
        
        
        # Lag the two characteristics by lagged_periods so we can apply the 
        # sort() function without lagged periods. Account for calibrate_cols.
        # -------------------------------------------------------------------
        
        # Name of the lagged firm characteristic columns
        if np.abs(self.lagged_periods) > 0:
            firm_char1 = (
                '%s_%s%d' %  
                (self.firm_characteristic,   
                 self.Tarrow,   
                 self.lagged_periods)
                )
        else:
            firm_char1 = self.firm_characteristic
        
        if np.abs(self.lagged_periods_2) > 0:
            firm_char2 = (
                '%s_%s%d' %  
                (self.firm_characteristic_2, 
                 self.Tarrow_2, 
                 self.lagged_periods_2)
                )
        else:
            firm_char2 = self.firm_characteristic_2
        # As a list
        firm_chars = [firm_char1, firm_char2]
        # Create a copy of the original dataframe to be used for 
        # double sorting (_ds stands for double sorting) 
        df_ds = self.df.copy()
        # Apply the lagged_periods operator for firm_characteristic
        df_ds[firm_char1] = (df_ds
                             .groupby(self.entity_id)[self.firm_characteristic]
                             .shift(self.lagged_periods)
                             )
        # Apply the lagged_periods_2 operator for firm_characteristic_2
        df_ds[firm_char2] = (df_ds
                           .groupby(self.entity_id)[self.firm_characteristic_2]
                           .shift(self.lagged_periods_2)
                             )
        # Drop rows for null values in firm_chars and calibrate_cols
        # Dataframe to be used in double sorting
        if self.calibrate_cols is not None:
            df_ds = df_ds.dropna(subset = firm_chars + self.calibrate_cols)
        else:
            df_ds = df_ds.dropna(subset = firm_chars)
        

        # --------------------------------
        # SORTING THE FIRST CHARACTERISTIC
        # --------------------------------
        
        # Single sort on firm_char1 with lagged_periods = 0 on df_ds
        single_sorted = self.sort(df_ds,
                                  firm_characteristic = firm_char1, 
                                  lagged_periods = 0, \
                                  n_portfolios = self.n_portfolios, 
                                  quantile_filter = self.quantile_filter, 
                                  save_sort = False)

            
        # ---------------------------------
        # SORTING THE SECOND CHARACTERISTIC
        # ---------------------------------
        
        # Second characteristic is dependent on the first.
        if self.conditional:
                                    
            # Double sort on firm_char2 with lagged periods = 0 on 
            # single_sorted dataframe
            double_sorted = (
                             single_sorted
                             .groupby(self.portfolio)
                             .apply(lambda x:  self.sort(x, 
                             firm_characteristic = firm_char2, 
                             lagged_periods = 0, 
                             n_portfolios = self.n_portfolios_2, 
                             quantile_filter = self.quantile_filter_2,
                             save_sort = False))
                             )
            # Reset the index
            double_sorted.reset_index(drop = True, inplace = True)
            # Define the double sort portfolio column
            double_sorted['Double_sort_portfolio'] = (
                double_sorted[self.portfolio].astype(int).astype(str) 
                + '_' 
                + double_sorted[self.portfolio_2].astype(int).astype(str)  
                )                 


        # Second characteristic is independent on the first.
        else:
            
            # Apply again self.Sort() 
            double_sorted = self.sort(single_sorted, 
                                      firm_characteristic = firm_char2, 
                                      lagged_periods = 0, 
                                      n_portfolios = self.n_portfolios_2, 
                                      quantile_filter = self.quantile_filter_2, 
                                      save_sort = False) 
                
            # Reset the index
            double_sorted.reset_index(drop = True, inplace = True)
            # Define the double sort portfolio column
            double_sorted['Double_sort_portfolio'] = (
                double_sorted[self.portfolio].astype(int).astype(str) 
                + '_' 
                + double_sorted[self.portfolio_2].astype(int).astype(str)  
                )
                

        # Define double_sorted and sort again by entity and time.
        self.double_sorted = (
                            double_sorted
                            .sort_values(by=[self.entity_id, self.time_id])
                            .reset_index(drop = True)
                            )
                
        # Save results
        if self.save_DoubleSort:
            filename = '%s.csv' % folder_name
            fpath = os.path.join(self.save_folder, filename)
            self.double_sorted.to_csv(fpath, index = False)        
        

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #      triple_sort           #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    # Triple Sorting
    def triple_sort(self, 
                   firm_characteristics, 
                   lagged_periods,
                   n_portfolios, 
                   quantile_filters = [None, None, None], 
                   conditional = [False, False], 
                   calibrate_cols = None, 
                   save_TripleSort = False):
        
        """
        Sort entities based on three characteristics. 
        
        Parameters:
        -----------
        firm_characteristics : list of str
           List elements are firm characteristics.
        lagged_periods : list of int
            List elements are the lagged perios of the firm characteristics.
        n_portfolios : list of int or numpy.array
            List elements are the portfolio quantiles for sorting.
        quantile_filters : list of list
            List elements are the quantile filters for the characteristics.           
        conditional : list of bool, default [False, False]
            It is a list of boolean values. Let A, B, C be the three firm 
            characteristics and '+' and '|' denote intersection and 
            conditionality of sets, respectively. 
            Then the interpretation of condtional is the following:
                [True, True]   = C|B|A
                [True, False]  = (C+B)|A
                [False, True]  = C|(A+B)
                [False, False] = A+B+C
        calibrate_cols : list, optional
            Only entities that have non-null values of calibrate_cols, 
            are sorted based on their firm_characteristics. We restrict 
            the set of characteristics that need to be available for an entity, 
            as the union of calibrate_cols and firm_characteristics.
        save_TripleSort : boolean, default True
            If True, the triple_sorted dataframe is saved in save_folder.

        Returns:
        --------
        triple_sorted : DataFrame
            DataFrame in which entities/stocks have been triple sorted 
            conditionally or unconditionally.        
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
            self.portfolio = (
                '%s_%s%d_portfolio' % 
                (self.firm_characteristic, 
                 self.Tarrow, 
                 np.abs(self.lagged_periods))
                )
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
            self.portfolio_2 = (
                '%s_%s%d_portfolio' %
                (self.firm_characteristic_2, 
                 self.Tarrow_2, 
                 np.abs(self.lagged_periods_2))
                )
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
            self.portfolio_3 = (
                '%s_%s%d_portfolio' % 
                (self.firm_characteristic_3,
                 self.Tarrow_3, 
                 np.abs(self.lagged_periods_3))
                )
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
        self.conditional = conditional
        # Differentiate between conditional and unconditional sorts.
        # 'D' stands for dependent/conditional
        # 'I' stands for indepedent/unconditional
        if self.conditional[0]:
            self.c1 = 'D'
        else:
            self.c1 = 'I'
        if self.conditional[1]:
            self.c2 = 'D'
        else:
            self.c2 = 'I'
        # Define the save_folder 
        folder_name = (
            '%dx%dx%d_%sx%s_portfolios_SortedBy_%sand%sand%s' % 
            (self.num_portfolios, 
             self.num_portfolios_2, 
             self.num_portfolios_3, 
             self.c1, 
             self.c2, 
             self.firm_characteristic, 
             self.firm_characteristic_2,
             self.firm_characteristic_3)
            )
        self.save_folder = os.path.join(self.save_dir, folder_name)
        if folder_name not in os.listdir(self.save_dir):
            os.mkdir(self.save_folder)
        self.save_TripleSort = save_TripleSort 
        
        
        
        # Function that combines two double sorted portfolio 
        # and defines the (C+B)|A case
        def combineTF(x, y):
            # Isolate the first sort on A 
            first_x = x.split('_')[0]
            first_y = y.split('_')[0]
            # They have to be the same. In our case they will always be.
            if first_x == first_y:
                # Isolate the second sorts of B|A and C|A
                second_x = x.split('_')[1]
                second_y = y.split('_')[1]
                # define the triple sort (C+B)|A
                return first_x+'_'+second_x+'_'+second_y
        
        
        # Lag the three characteristics by lagged_periods so we can apply 
        # the Sort() function without lagged periods. 
        # -----------------------------------------------------------------
        
        # Name of the lagged firm characteristic columns
        if np.abs(self.lagged_periods) > 0:
            firm_char1 = (
                '%s_%s%d' %  
                (self.firm_characteristic,  
                 self.Tarrow,   
                 self.lagged_periods)
                )
        else:
            firm_char1 = self.firm_characteristic
            
        if np.abs(self.lagged_periods_2):
            firm_char2 = (
                '%s_%s%d' % 
                (self.firm_characteristic_2, 
                 self.Tarrow_2, 
                 self.lagged_periods_2)
                )
        else:
            firm_char2 = self.firm_characteristic_2
            
        if np.abs(self.lagged_periods_3):
            firm_char3 = (
                '%s_%s%d' %  
                (self.firm_characteristic_3, 
                 self.Tarrow_3, 
                 self.lagged_periods_3)
                )
        else:
            firm_char3 = self.firm_characteristic_3
            
        # As a list
        firm_chars = [firm_char1, firm_char2, firm_char3]
        # Create a copy of the original dataframe to be used for triple sorting (_ts stands for triple sorting)
        df_ts = self.df.copy()
        # Apply the lagged_periods operator for firm_characteristic
        df_ts[firm_char1] = (
                            df_ts
                            .groupby(self.entity_id)[self.firm_characteristic]
                            .shift(self.lagged_periods)
                            )
        # Apply the lagged_periods_2 operator for firm_characteristic_2
        df_ts[firm_char2] = (
                           df_ts
                           .groupby(self.entity_id)[self.firm_characteristic_2]
                           .shift(self.lagged_periods_2)
                            )
         # Apply the lagged_periods_3 operator for firm_characteristic_3
        df_ts[firm_char3] = (
                           df_ts
                           .groupby(self.entity_id)[self.firm_characteristic_3]
                           .shift(self.lagged_periods_3)
                            )
        
        # Drop rows for null values in firm_chars
        # Dataframe to be used in triple sorting
        if self.calibrate_cols is not None:
            df_ts = df_ts.dropna(subset=firm_chars + self.calibrate_cols)
        else:
            df_ts = df_ts.dropna(subset=firm_chars)
            
   
         

        # --------------------------------
        # SORTING THE FIRST CHARACTERISTIC
        # --------------------------------
        
        # Single sort on firm_char1 with lagged_periods = 0 on df_ts
        single_sorted = self.sort(df_ts, 
                                  firm_characteristic = firm_char1,
                                  lagged_periods = 0, 
                                  n_portfolios = self.n_portfolios,
                                  quantile_filter = self.quantile_filter,
                                  save_sort = False)
            
        # ---------------------------------
        # SORTING THE SECOND CHARACTERISTIC
        # ---------------------------------       
        
        # First boolean value is True
        if self.conditional[0]:
            
            # Double sort on firm_char2 with lagged periods = 0 on single_sorted dataframe
            double_sorted = (
                        single_sorted
                        .groupby(self.portfolio)
                        .apply(lambda x:  self.sort(x, 
                                    firm_characteristic = firm_char2, 
                                    lagged_periods = 0, \
                                    n_portfolios = self.n_portfolios_2,
                                    quantile_filter = self.quantile_filter_2, 
                                    save_sort = False))   
                            )
                
            # Reset the index of double_sorted dataframe
            double_sorted.reset_index(drop=True, inplace = True)
            
            # Define the double sort portfolio column
            double_sorted['Double_sort_portfolio'] = (
                double_sorted[self.portfolio].astype(int).astype(str) 
                + '_' 
                + double_sorted[self.portfolio_2].astype(int).astype(str)  
                )

        # First boolean value of False            
        else:
            
            double_sorted = self.sort(single_sorted,
                                      firm_characteristic = firm_char2, 
                                      lagged_periods = 0, 
                                      n_portfolios = self.n_portfolios_2,
                                      quantile_filter = self.quantile_filter_2, 
                                      save_sort = False)
                            
            # Define the double sort portfolio column
            double_sorted['Double_sort_portfolio'] = (
                double_sorted[self.portfolio].astype(int).astype(str) 
                + '_' 
                + double_sorted[self.portfolio_2].astype(int).astype(str) 
                )
            
            

        # --------------------------------
        # SORTING THE THIRD CHARACTERISTIC
        # --------------------------------      
        
        # Second boolean value is True
        if self.conditional[1]:    
            
            # Cases of [True, True] = C|B|A or [False, True] = C|(A+B)
            
            # Triple sort on firm_char3 with lagged periods = 0 on double_sorted dataframe
            triple_sorted = (
                        double_sorted
                        .groupby('Double_sort_portfolio')
                        .apply(lambda x:  self.sort(x, 
                                    firm_characteristic = firm_char3,
                                    lagged_periods = 0, 
                                    n_portfolios = self.n_portfolios_3, 
                                    quantile_filter = self.quantile_filter_3, 
                                    save_sort = False))
                        )
            # Reset the index
            triple_sorted.reset_index(drop = True, inplace = True)
            
            # Define the triple sort portfolio column
            triple_sorted['Triple_sort_portfolio'] = (
                triple_sorted['Double_sort_portfolio'] 
                + '_'
                + triple_sorted[self.portfolio_3].astype(int).astype(str)
                )
                             
        # Second boolean value is False
        else:
            
            # Case of [True, False] = (C+B)|A
            if self.conditional[0]:
                
                
                # The True-False case is tricky, because it corresponds to the 
                # creation of intersection of sets of characteristics B and C 
                # after conditioning on A; (C+B)|A or C|A + B|A.
                # Thus the operation should not be on the double sort portfolio 
                # but rather first on A to create C|A and another 
                # 'Double_sort_portfolio_2' column. Then we need to combine
                # the two double sort portfolio columns:
                # 'Double_sort_portfolio' -> B|A and 
                # 'Double_sort_portfolio' -> C|A
                # to get the true triple sort portfolio for the configuration 
                # (C+B)|A. 
                
            
                # C|A 
                # Double sort on firm_char3 with lagged periods = 0 on single_sorted dataframe
                triple_sorted = (
                        double_sorted
                        .groupby(self.portfolio)
                        .apply(lambda x:  self.sort(x, 
                                    firm_characteristic = firm_char3,
                                    lagged_periods = 0, 
                                    n_portfolios = self.n_portfolios_3,
                                    quantile_filter = self.quantile_filter_3, 
                                    save_sort = False)) 
                        )
                    
                # Reset the index of triple_sorted dataframe
                triple_sorted.reset_index(drop=True, inplace = True)
                # Define the second double sort portfolio column
                triple_sorted['Double_sort_portfolio_2'] = (
                    triple_sorted[self.portfolio].astype(int).astype(str) 
                    + '_' 
                    + triple_sorted[self.portfolio_3].astype(int).astype(str)
                    )
        
                          
                # B|A + C|A 
                # Define the triple sort portfolio column (apply lambda cannot be avoided)
                triple_sorted['Triple_sort_portfolio'] = (
                    triple_sorted
                    .apply(lambda x: combineTF(x['Double_sort_portfolio'],
                                               x['Double_sort_portfolio_2']),
                           axis = 1)
                    )
                
            
            # Case of [False, False] = A+B+C
            else:
                
                # Independent sort
                triple_sorted = self.sort(double_sorted, 
                                          firm_characteristic = firm_char3, 
                                          lagged_periods = 0, 
                                          n_portfolios = self.n_portfolios_3, 
                                          quantile_filter = self.quantile_filter_3, 
                                          save_sort = False)
                # Define the triple sort portfolio column
                triple_sorted['Triple_sort_portfolio'] = (
                    triple_sorted['Double_sort_portfolio'] 
                    + '_' 
                    + triple_sorted[self.portfolio_3].astype(int).astype(str)
                    )

                    
        # Define triple_sorted and sort again by entity and time.
        self.triple_sorted = (
                            triple_sorted
                            .sort_values(by=[self.entity_id, self.time_id])
                            .reset_index(drop = True)
                            )
        
        
        # Save results
        if self.save_TripleSort:
            filename = '%s.csv' % folder_name    
            fpath = os.path.join(self.save_folder, filename)                                                                
            self.triple_sorted.to_csv(fpath, index = False)
            

    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #      ff_portfolios         #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
    # Function that generates the weighted average returns of portfolios 
    # of entities that have been sorted by their characteristics. 
    def ff_portfolios(self, 
                     ret_data, 
                     ret_time_id, 
                     ff_characteristics, 
                     ff_lagged_periods, 
                     ff_n_portfolios,
                     ff_quantile_filters = None, 
                     ff_conditional = None, 
                     weight_col = None, 
                     return_col = 'RET', 
                     market_cap_cols = [], 
                     ff_dir = None, 
                     ff_save = False):
        """
        Sort entities into portfolios and calculate weighted-average returns.
        
        Parameters:
        -----------
        ret_data : Dataframe
            Dataframe where returns for entities are stored in a panel format.
        ret_time_id : str
            Time identifier as found in ret_data. ret_time_id dictates the
            frequency for which the portfolio returns are calculated.
        ff_characteristics : list of str
            A list of up to three characteristics for which entities will be 
            sorted.
        ff_lagged_periods : list of int
            A list of the number of lagged periods for the characteristics
            to be sorted. The length of characteristics and lagged_periods 
            must match.
        ff_n_portfolios : list of int or numpy.array
            Integer values correspond to N_portfolios equal portfolios.
            numpy.array values correspond to quantiles. 
        ff_quantile_filters : list of list, optional
            Each element corresponds to filtering entities for the ranking of 
            portfolios into each firm characteristic. The length of the list 
            must be the same as that of firm_characteristics.
        ff_conditional: list of bool, optional
            It is a list of boolean values. Let A, B, C be the three
            firm characteristics and '+' and '|' denote intersection and
            conditionality of sets, respectively. 
            Then the interpretation of condtional is the following:
                [True, True]   = C|B|A
                [True, False]  = (C+B)|A
                [False, True]  = C|(A+B)
                [False, False] = A+B+C
        weight_col : str, optional
            The column used for weighting the returns in a portfolio. 
            If weight_col is None, the portfolios are equal-weighted.
        return_col : str, default 'RET'
            The column of ret_data that corresponds to returns. 
            Default value is 'RET' which is the name of returns from CRSP.
        market_cap_cols : list of str, default []
            A list with two elements:
                i. First element: start of the current period market 
                capitalization of the entity
                ii. Second element : end of the current period market
                capitalization of the entity
            If this list is not empty, then the turnover of the 
            portfolios will be calculated. The current and the previous
            end-of-period market cap will be used to get the total return of 
            the stock so that end-of-period weights of the portfolio strategy 
            can be calculated and incorporated in the turnover formula.
        ff_dir : dir, optional
            Saving directory.
        ff_save : boolean, default True
            If True, save results to FFdir.
            
        Returns:
        --------
        portfolios : DataFrame
            Dataframe with columns = portfolios, index = ret_time_id, 
            values = returns
        num_stocks : DataFrame
            Dataframe with columns = portfolios, index = ret_time_id, 
            values = number of stocks in each portfolio
        ff_class : PortSort class
        turnover : Dataframe
            Dataframe with columns = portfolios, index = time_id, 
            values = portfolio turnover
        turnover_raw : DataFrame
            Raw turnover DataFrame       
        """
        
        # Instance variables
        self.ret_data = ret_data
        self.ret_time_id = ret_time_id
        self.ff_characteristics = ff_characteristics
        self.ff_lagged_periods = ff_lagged_periods
        self.ff_n_portfolios = ff_n_portfolios
        self.ff_quantile_filters = [None]*len(self.ff_characteristics)
        if ff_conditional is None and len(ff_characteristics)>1:
            self.ff_conditional = [False]*(len(self.ff_characteristics)-1)
        else:
            self.ff_conditional = ff_conditional
        self.weight_col = weight_col
        self.return_col = return_col
        self.market_cap_cols = market_cap_cols
        self.ff_dir = ff_dir if ff_dir is not None else self.save_dir
        self.ff_save = ff_save
        
        
        # Create a copy of the original FirmCharacteristics dataframe
        firmchars = self.df.copy()
        
        # Create a new PortSort class with df = firmchars
        ff_class = PortSort(df=firmchars, 
                           entity_id = self.entity_id, 
                           time_id = self.time_id, 
                           prefix_name = self.prefix_name, 
                           save_dir = self.save_dir)
        """
        CALIBRATING 
        -----------
        weight_col as the calibration column if not None
        return_col is added to the calibration columns only when the
        portfolio rebalancing and the stock retrun frequency are the same.
        Of course, the return_col has to be in the columns of firmchars.
        """
        
        # Portfolio rebalancing and characteristics have the same frequency
        if self.time_id == self.ret_time_id:            
            if self.weight_col is not None: 
                if self.return_col in firmchars:
                    ff_calibrate_col = [self.weight_col, self.return_col]
                else:
                    ff_calibrate_col = [self.weight_col]
            # weight_col is None
            else:
                if self.return_col in firmchars:
                    ff_calibrate_col = [self.return_col]
                else:
                    ff_calibrate_col = None
        # Portfolio rebalancing and characteristics do NOT have the 
        # same frequency
        else:
            if self.weight_col is not None: 
                ff_calibrate_col = [self.weight_col]
            else:
                ff_calibrate_col = None
                


        # -----------------------------------
        #  SORT -- SINGLE or DOUBLE or TRIPLE
        # -----------------------------------
        
        
        # One characteristic --> Single Sort
        # ----------------------------------
        if len(self.ff_characteristics) == 1:
            
             
            # Single sort
            ff_class.single_sort(firm_characteristic=self.ff_characteristics[0], 
                               n_portfolios = self.ff_n_portfolios[0], 
                               lagged_periods = self.ff_lagged_periods[0],
                               quantile_filter = self.ff_quantile_filters[0],
                               calibrate_cols = ff_calibrate_col, 
                               save_SingleSort = self.ff_save) 
              
            # Name of the single sorted portfolio
            port_name = ff_class.portfolio
            
                                                
            # Isolate only the essential columns for portfolio assignment.
            # Include or not weighting column
            if self.weight_col is not None:
                iso_cols = [self.time_id, self.entity_id, 
                            self.weight_col, port_name]
                   
            else:
                iso_cols = [self.time_id, self.entity_id, port_name]             
            ports = ff_class.single_sorted[iso_cols].copy() 
            
            # Define save names
            save_str =  (
                '%d_portfolios_sortedBy_%s.csv' % 
                (ff_class.num_portfolios, self.ff_characteristics[0])
                )
            save_ret = 'RET_' + save_str
            save_num = 'NUM_STOCKS_' + save_str
            
        # Two characteristic --> Double Sort
        # -----------------------------------
        if len(self.ff_characteristics) == 2:
                   
            
            # Double sort
            ff_class.double_sort(firm_characteristics=self.ff_characteristics,
                               lagged_periods = self.ff_lagged_periods,
                               n_portfolios = self.ff_n_portfolios, 
                               quantile_filters = self.ff_quantile_filters, 
                               conditional = self.ff_conditional[0], 
                               calibrate_cols = ff_calibrate_col, 
                               save_DoubleSort = self.ff_save)   
            
            # Isolate only the essential columns for portfolio assignment
            port_name = 'Double_sort_portfolio'
            # Include or not weighting column
            if self.weight_col is not None:
                iso_cols = [self.time_id, self.entity_id, 
                            self.weight_col, port_name]
            else:
                iso_cols = [self.time_id, self.entity_id, port_name]
            ports = ff_class.double_sorted[iso_cols].copy()
            # Define save names
            save_str =  (
                '%dx%d_portfolios_sortedBy_%sand%s.csv' % 
                (ff_class.num_portfolios, 
                 ff_class.num_portfolios_2, 
                 self.ff_characteristics[0], 
                 self.ff_characteristics[1])
                )
            save_ret = 'RET_' + save_str
            save_num = 'NUM_STOCKS_' + save_str
            
        
        # Three characteristics --> Triple Sort
        # --------------------------------------
        if len(self.ff_characteristics) == 3:
            
            
            # Triple sort
            ff_class.triple_sort(firm_characteristics=self.ff_characteristics,
                               lagged_periods = self.ff_lagged_periods, 
                               n_portfolios = self.ff_n_portfolios,
                               quantile_filters = self.ff_quantile_filters, 
                               conditional = self.ff_conditional,
                               calibrate_cols = ff_calibrate_col, 
                               save_TripleSort = self.ff_save)
    
            # Isolate only the essential columns for portfolio assignment
            port_name = 'Triple_sort_portfolio'
            # Include or not weighting column
            if self.weight_col is not None:
                iso_cols = [self.time_id, self.entity_id, 
                            self.weight_col, port_name]  
            else:
                iso_cols = [self.time_id, self.entity_id, port_name] 
            ports = ff_class.triple_sorted[iso_cols].copy()
            
            # Define save names
            save_str =  (
                '%dx%dx%d_%sx%s_portfolios_sortedBy_%sand%sand%s.csv' % 
                (ff_class.num_portfolios, 
                ff_class.num_portfolios_2, 
                ff_class.num_portfolios_3, 
                ff_class.c1, ff_class.c2, 
                self.ff_characteristics[0], 
                self.ff_characteristics[1],
                self.ff_characteristics[2])
                    )
            save_ret = 'RET_' + save_str
            save_num = 'NUM_STOCKS_' + save_str
                



        # --------
        # TURNOVER 
        # --------
        
        """
        Turnover definition
        -------------------
        
        I use the portfolio turnover definition of equation 13 of 
        DeMiguel et al. (2009). The paper can be found at:
        https://pubsonline.informs.org/doi/10.1287/mnsc.1080.0986    
        
        I assume that the turnover of period t corresponds to the construction 
        of the portfolio of the next period t+1. Thus I rebalance at the end of 
        period t before going to period t+1. 
        """
        
        # Check if market_cap_cols is empty
        if len(market_cap_cols) == 0:
            # No turnover is calculated
            self.turnover = None
            self.turnover_raw = None
        # Check if market_cap_cols has the correct information
        if len(market_cap_cols) == 2:
            
            # Start of the period cap
            cap_start = market_cap_cols[0]
            self.cap_start = cap_start
            # End of the period cap
            cap_end = market_cap_cols[1]
            self.cap_end = cap_end
            
            """
            STEP 0 : Isolate only the essential columns
            -------------------------------------------
            We do not need all the columns of the characteristic sorted 
            dataframe resulted from the definition of the FFclass. We only 
            need to isolate the time-entity pair, the weight column (if any)
            and the market_cap_cols
            """
            
            # Union of weight column and market capitilization columns
            turnover_cols = list(set( [self.weight_col] + market_cap_cols ))
            # Columns to isolate from sorted DataFrame with or without 
            # weight_col.
            withW_cols = [self.time_id, self.entity_id, port_name] + turnover_cols
            noW_cols = [self.time_id, self.entity_id, port_name] + market_cap_cols
            
            
            if self.weight_col is not None:
                if len(self.ff_characteristics) == 1:
                    df_s = ff_class.single_sorted[withW_cols].copy()
                if len(self.ff_characteristics) == 2:
                    df_s = ff_class.double_sorted[withW_cols].copy()
                if len(self.ff_characteristics) == 3:
                    df_s = ff_class.triple_sorted[withW_cols].copy()
            else:            
                if len(self.ff_characteristics) == 1:
                    df_s = ff_class.single_sorted[noW_cols].copy()
                if len(self.ff_characteristics) == 2:
                    df_s = ff_class.double_sorted[noW_cols].copy()
                if len(self.ff_characteristics) == 3:
                    df_s = ff_class.triple_sorted[noW_cols].copy()

            
            


            """
            STEP 1 : Portfolio columns for holdings 
            -----------------------------------------
            The current and next period holdings of the constructed portfolios 
            are tracked for each period at the entity level. Two sets of 
            columns are created. The first set is the current period holdings
            denoted by the names of the portfolios and it has a value of 1 if 
            an entity belongs to that portfolio and 0 otherwise. The second set
            is the next period holdings denoted by the names of the portfolios
            augmented with the suffix '_for1' (forward one period ahead) and 
            it has the value of 1 if an entity belongs to the portfolio in the 
            next period and 0 otherwise.
            """
            
            portfolio_cols = list(
                df_s[port_name].value_counts().sort_index().index.values
                )
            # Convert to string
            if len(ff_characteristics) == 1:
                portfolio_cols = [ str(int(x)) for x in portfolio_cols]
            else:
                portfolio_cols = [ x for x in portfolio_cols]            
            
            # Define the portfolio holding columns
            for x in portfolio_cols:
                # End of period portfolio holding
                if len(ff_characteristics) == 1:      
                    df_s[x] = (
                                df_s[port_name]
                                .apply(lambda y: 1 if y == float(x) else 0)
                                )
                else:
                    df_s[x] = (
                                df_s[port_name]
                                .apply(lambda y: 1 if y == x else 0) 
                                )
                # Next period portfolio holding
                df_s[x+'_for1'] = (
                                    df_s
                                    .groupby(self.entity_id)[x]
                                    .shift(-1)
                                    .fillna(value = 0)  
                                    )
       
            """
            STEP 2 : Old and new weights in a period t    
            ------------------------------------------
            First, the entity weights of each portfolio at the start of each 
            period t are explicitly computed and denoted as 'Start_weights'.
            Then, the current and previous end-of-period t market 
            capitalization of the entity as in market_cap_cols, are used 
            to construct the 'Old_weights' as 'Start_weights'
            *( current end-of-period cap)/(previous end-of-period cap). 
            The 'Old_weights' column contains the weights in the portfolios 
            after the period t has passed and these are the weigths we need to
            rebalance to hold the portfolio for the next period t+1. 
            The 'New_weights' are the one-period lagged 'Start_weights' of
            period t+1 that are going to be used to construct the portfolio 
            strategy at then end of period t for the duration of period t+1(. 
            """

              
            # Not equal-weighting scheme
            if self.weight_col is not None:
                # Start_weights : weights at the start of the period t
                df_s['Start_weights'] = (
                        df_s
                        .groupby([port_name, self.time_id])[self.weight_col]
                        .transform(lambda x: x/x.sum() )
                        )
                # Old weights : weights at the end of the period t before rebalancing
                df_s['Old_weights_raw'] =  df_s['Start_weights']*df_s[cap_end]/df_s[cap_start]
                df_s['Old_weights_raw'].fillna(value = 0, inplace = True)
                # Normalize
                df_s['Old_weights'] = (
                        df_s
                        .groupby([port_name, self.time_id])['Old_weights_raw']
                        .transform(lambda x: x/x.sum())
                        )
                # New weights : weights at the end of the period t after rebalancing
                df_s['New_weights'] = (
                    df_s.groupby(self.entity_id)['Start_weights'].shift(-1)
                    )
                df_s['New_weights'].fillna(value = 0, inplace = True)
            # Equal-weighting scheme
            else:
                # Start_weights : weights at the start of the period t
                df_s['Start_weights'] = (
                        df_s
                        .groupby([port_name, self.time_id])[self.entity_id]
                        .transform(lambda x: 1/x.count() )
                        )
                # Old weights : weights at the end of the period t before rebalancing
                df_s['Old_weights_raw'] =  (
                        df_s['Start_weights']*df_s[cap_end]/df_s[cap_start]
                    )
                df_s['Old_weights_raw'].fillna(value = 0, inplace = True)
                # Normalize
                df_s['Old_weights'] = (
                        df_s
                        .groupby([port_name, self.time_id])['Old_weights_raw']
                        .transform(lambda x: x/x.sum())
                        )
                # New weights : weights at the end of the period t after rebalancing
                df_s['New_weights'] = (
                    df_s.groupby(self.entity_id)['Start_weights'].shift(-1)
                    )
                df_s['New_weights'].fillna(value = 0, inplace = True)
                
                


            """
            STEP 3 - Weight change from period t to the next period t+1
            ------------------------------------------------------------
            We now track the weight change of each entity in the portfolios 
            of FFclass. The formula is 
            New_weights*(current period t portfolio holding) 
            - Old_weights*(next period t+1 portfolio holding).
            There are 3 distinct cases:
                1. Entity i stays in the same portfolio from period t to t+1: 
                    w_{t+1} - w_{t+}
                2. Entity i is added in the portfolio from period t to t+1:
                    w_{t+1} - 0 = w_{t+1}
                3. Entity i is removed from the portfolio in period t+1:
                    0 - w_{t+} = - w_{t+}
            """
                        
            # Define the weight change of an entity from end of period t to 
            # the start of period t+1
            for x in portfolio_cols:
                df_s[x+'_dWeight'] = (
                                        df_s['New_weights']*df_s[x+'_for1'] 
                                        - df_s['Old_weights']*df_s[x]
                                        )
                                    


            """
            STEP 4 - Turnover dataframe
            ----------------------------
            Define the portfolio turnover dataframe  of period t as the sum 
            of the absolute value of the 'dWeight' columns.
            """
            
            turnover = (
                df_s
                .groupby(by = self.time_id)[[x+'_dWeight' for x in portfolio_cols]]
                .apply(lambda x: x.abs().sum())   
                )
            # Rename the columns
            turnover.columns = [x.replace('_dWeight', '') for x in turnover.columns]
            self.turnover = turnover    
            # Raw turnover dataframe
            self.turnover_raw = df_s
        
        
        # Number of stocks in a portfolio
        # -------------------------------
        num_stocks = (
                        ports
                        .groupby(by = [port_name, self.time_id] )[port_name]
                        .count()
                        .unstack(level=0)
                        )        
        self.num_stocks = num_stocks
        
        
        # Portfolio weighted-average returns
        # ----------------------------------
        
        # The inner merging is taking care of stocks that should be excluded
        # from the formation of the portfolios.
        ret_ports = pd.merge(self.ret_data, 
                             ports, 
                             how='inner', 
                             on=[self.time_id, self.entity_id], 
                             suffixes = ('', '_2'))

        
        # Weighted average portfolio returns
        if self.weight_col is None:
            char_ports = (
                ret_ports
                .groupby([port_name, self.ret_time_id])[self.return_col]
                .mean()
                .unstack(level=0)
                )
        else:
            # Define the proper weights using weight_col
            ret_ports['proper_W'] = (
                        ret_ports                
                        .groupby([port_name, self.time_id])[self.weight_col]
                        .transform(lambda x: x/x.sum() )
                        )
            # Multiply returns with proper weights
            ret_ports['RET*W'] = ret_ports[self.return_col]*ret_ports['proper_W']
            char_ports = (
                ret_ports
                .groupby([port_name, self.ret_time_id])['RET*W']
                .sum()
                .unstack(level=0)
                )
                       
        self.portfolios = char_ports
            
        
        # Return the class
        self.ff_class = ff_class

        
        #-------------
        # SAVE RESULTS
        # ------------
                
        char_ports.to_csv(os.path.join(ff_class.save_folder, save_ret ))
        num_stocks.to_csv(os.path.join(ff_class.save_folder, save_num ))
        
        

        

        
    

      
        




        
        
        


