import pandas as pd

def keep_regs(df, regs):
    """Keep only the subset regs of regions in data.

    Args:
        df (pd.DataFrame): pandas dataframe 
        regs (list): list of regions to keep

    Returns:
        df (pd.DataFrame): filtered pandas dataframe
    """ 
    for r in regs:
        I = df.reg.str.contains(r)
        df = df.loc[I == False] # keep everything else
    return df

def define_params():
    """Define the parameters for fetching data.
    
    Returns:
        dict: parameter dictionary
    """
    params = {'table': 'aul01',
              'format': 'BULK',
              'lang': 'en',
              'variables': [{'code': 'OMRÅDE', 'values': ['000']},
                           {'code': 'YDELSESTYPE', 'values': ['TOT']},
                           {'code': 'AKASSE', 'values': ['*']},
                           {'code': 'ALDER', 'values': ['TOT']},
                           {'code': 'KØN', 'values': ['TOT']},
                           {'code': 'Tid', 'values': ['*']}]}
    return params

def fetch_data(unemp, params):
    """Fetch data using the provided parameters.
    
    Args:
        unemp (DstApi): unemployment data API instance
        params (dict): parameter dictionary
        
    Returns:
        pd.DataFrame: fetched data
    """
    return unemp.get_data(params=params)

def drop_columns(df, columns):
    """Drop specified columns from dataframe.
    
    Args:
        df (pd.DataFrame): pandas dataframe
        columns (list): list of columns to drop
        
    Returns:
        pd.DataFrame: dataframe with dropped columns
    """
    df.drop(columns, axis=1, inplace=True)
    return df

def get_unique_akasse(df):
    """Get unique unemployment insurance funds.
    
    Args:
        df (pd.DataFrame): pandas dataframe
        
    Returns:
        np.ndarray: array of unique unemployment insurance funds
    """
    return df['AKASSE'].unique()

def analyze_unemployment(df):
    """Analyze unemployment over the years.
    
    Args:
        df (pd.DataFrame): pandas dataframe
        
    Returns:
        pd.DataFrame: dataframe with unemployment analysis
    """
    return df[(df['AKASSE'] == 'Total')][['TID', 'INDHOLD']]
