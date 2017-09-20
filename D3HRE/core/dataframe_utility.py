from numpy import floor

def full_day_cut(df):
    '''
    Crop dataFrame at the end of the day

    :param df: pandas data frame
    :return: pandas data frame that end at full day
    '''
    df = df[0:int(floor(len(df) / 24)) * 24]
    return df