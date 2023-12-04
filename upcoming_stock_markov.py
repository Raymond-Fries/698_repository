import numpy as np
import pandas as pd
from random import seed
import psycopg2
import os
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from sklearn.preprocessing import PowerTransformer
from data_generation import get_all_holidays_csv,get_all_fed_dates_csv,get_all_er_dates_csv


def get_historical_df(symbol,dayofweek):
    df = pd.read_csv('Data/Daily_Intraday/Daily_Means/'+symbol.upper()+'_'+str(dayofweek)+'_daily_minute_means.csv',header=0)
    df = df.drop(['Unnamed: 0'],axis=1)
    return df

def get_close_over_close_boundaries(symbol,dayofweek,stds):
    df = pd.read_csv('Data/Garch_Results/'+symbol+'_'+str(dayofweek)+'_'+str(stds)+'stds_Close_over_Close_Garch_Results.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop('Unnamed: 0',axis=1)
    return df

def get_open_boundaries(symbol,dayofweek,stds):
    df = pd.read_csv('Data/Garch_Results/'+symbol+'_'+str(dayofweek)+'_'+str(stds)+'stds_Intraday_Open_Garch_Results.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop('Unnamed: 0',axis=1)
    return df

def get_mean_boundaries(symbol,dayofweek,stds):
    df = pd.read_csv('Data/Garch_Results/'+symbol+'_'+str(dayofweek)+'_'+str(stds)+'stds_Intraday_Mean_Garch_Results.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop('Unnamed: 0',axis=1)
    return df

def get_std_boundaries(symbol,dayofweek,stds):
    df = pd.read_csv('Data/Garch_Results/'+symbol+'_'+str(dayofweek)+'_'+str(stds)+'stds_Intraday_Standard_Deviation_Garch_Results.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop('Unnamed: 0',axis=1)
    return df
def get_previous_close(symbol,last_date):
    df = pd.read_csv('Data/Daily_Intraday/Previous_Close/previous_close.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['dates'] = df['timestamp'].dt.date
    dates = list(df['dates'].unique())
    last_close_index = dates.index(datetime.strptime(last_date,"%Y-%m-%d %H:%M:%S").date())
    temp = df[(df['symbol'] == symbol) & (df['dates'] == dates[last_close_index-1])].reset_index()
    return temp['close'].iloc[0]

def truncate(df, decimals=0):
    multiplier = 10 ** decimals
    df = (df * multiplier) / multiplier
    return df

def truncate_to_even_cents(df,previous_close):
    df = df.cumsum(axis=0)
    df = previous_close + (df * previous_close)
    df = df.round(2)
    new_row = pd.Series([previous_close] * len(df.columns))
    temp = new_row.to_frame().T
    df = truncate(df,2)
    df2 = pd.concat([temp,df],axis=0).reset_index(drop=True)
    df = pd.DataFrame()
    temp= pd.DataFrame()
    df2 = df2.pct_change()
    df2 = df2.dropna(axis=0)
    return df2
def get_data(symbol,dayofweek,time):
    df = pd.read_csv('Data/Daily_Intraday/'+symbol.upper()+'_intraday.csv',header=0)
    df = df.drop(['Unnamed: 0'],axis=1)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df = df[(df['timestamp'].dt.dayofweek == dayofweek) & (df['timestamp'].dt.time == time)].reset_index()
    df = df.drop('index',axis=1)
    remove_h = get_all_holidays_csv()
    remove_f = get_all_fed_dates_csv()
    remove_e = get_all_er_dates_csv(symbol)
    no_phenom = list(set(remove_h+remove_f+remove_e))
    temp =df[~df['date'].isin(list(no_phenom))]
    temp = temp.drop('date',axis=1)
    df = pd.DataFrame()
    return temp

def get_general_data(symbol,dayofweek,time):
    df = pd.read_csv('Data/Daily_Intraday/'+symbol.upper()+'_intraday.csv',header=0)
    df = df.drop(['Unnamed: 0'],axis=1)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df = df[(df['timestamp'].dt.dayofweek == dayofweek) & (df['timestamp'].dt.time == time)].reset_index()
    df = df.drop('index',axis=1)
    remove_h = get_all_holidays_csv()
    remove_f = get_all_fed_dates_csv()
    remove_e = get_all_er_dates_csv(symbol)
    no_phenom = list(set(remove_h+remove_f+remove_e))
    temp =df[~df['date'].isin(list(no_phenom))]
    temp = temp.drop('date',axis=1)
    df = pd.DataFrame()
    return temp

def create_random_probs(symbol,dayofweek,stds):
    
    ccb = get_close_over_close_boundaries(symbol,dayofweek,stds)
    ob = get_open_boundaries(symbol,dayofweek,stds)
    mb = get_mean_boundaries(symbol,dayofweek,stds)
    sb = get_std_boundaries(symbol,dayofweek,stds)
    last_date = str(sb['Date'].iloc[0])
    previous_close = get_previous_close(symbol,last_date)
    mean_upper = mb['Upper_Bound'].iloc[0]
    mean_lower = mb['Lower_Bound'].iloc[0]
    open_upper = ob['Upper_Bound'].iloc[0]
    open_lower = ob['Lower_Bound'].iloc[0]
    std_upper = sb['Upper_Bound'].iloc[0]
    std_lower = sb['Lower_Bound'].iloc[0]
    eod_upper = ccb['Upper_Bound'].iloc[0]
    eod_lower = ccb['Lower_Bound'].iloc[0]
    print(symbol,' ',dayofweek,' ',stds,' ',std_upper,' ',std_lower,' ',mean_upper,' ',mean_lower)
    times = pd.date_range("09:30:00", "15:59:00", freq="1min").time
    count = 0
    count1 = 0
    while count < 1000:
        start = datetime.now()
        array = []
                    
        for time in times:
            
            distro = get_data(symbol,dayofweek,time)

            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            distro['trans'] = pt.fit_transform(np.array(distro['change']).reshape(len(distro['change']),1)).reshape(1,-1)[0]
            values = np.random.normal(loc=distro['trans'].mean(),scale=distro['trans'].std(),size=850000)
            #inverse yeo
            values = pt.inverse_transform(values.reshape(-1, 1)).reshape(1,-1)[0]
            
            array.append(values)
            values = []
            distro = pd.DataFrame()
        df = pd.DataFrame(array)
        array=[]
        count1 += 1
        cents = truncate_to_even_cents(df,previous_close)
        df = pd.DataFrame()
        
        most_likely = cents.loc[:,(cents.std() <= std_upper) & (cents.std() >= std_lower) & (cents.mean() <= mean_upper) & (cents.mean() >= mean_lower) & (cents.iloc[0] <= open_upper) & (cents.iloc[0] >= open_lower) & (cents.cumsum().iloc[389] <= eod_upper) & (cents.cumsum().iloc[389] >= eod_lower)]
        #most_likely = most_likely.drop('index',axis=1)
        
        cents = pd.DataFrame()
        print(symbol,' count1 ',count1,' ',dayofweek,' ',str(stds),' ',len(most_likely.columns),' ',most_likely.mean().min(),' ',most_likely.mean().max(),' ',most_likely.std().min(),' ',most_likely.std().max())
        if len(most_likely.columns) > 0:

            for i in most_likely.columns:
                #requirements_df[i].mean() <= mean_upper and requirements_df[i].mean() >= mean_lower and
                name = str(i)
                plt.plot(most_likely[i].cumsum(),alpha=.35,label=name+'_Simulation')
                count += 1
            print('count: ',count)
        most_likely = pd.DataFrame()
        print(datetime.now()-start)
    plt.title(symbol+' '+str(dayofweek)+' Simulation with '+str(stds)+' standard deviation for all constraints')
    #plt.savefig('Data/Simulations_Images/'+symbol+'_'+str(dayofweek)+'_'+str(stds)+'_No_Phenomena_Constraints_SVR-GARCH.png')
    plt.show()
    #plt.close()
    
def general_create_random_probs(symbol,dayofweek,stds):
    distro = get_historical_df(symbol,dayofweek)
    sb = get_std_boundaries(symbol,dayofweek,stds)
    last_date = str(sb['Date'].iloc[0])
    previous_close = get_previous_close(symbol,last_date)

    count = 0
    count1 = 0
    while count < 1000:
        times = pd.date_range("09:30:00", "15:59:00", freq="1min").time

        array = []
                    
        for time in times:
            
            distro = get_general_data(symbol,dayofweek,time)

            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            distro['trans'] = pt.fit_transform(np.array(distro['change']).reshape(len(distro['change']),1)).reshape(1,-1)[0]
            values = np.random.normal(loc=distro['trans'].mean(),scale=distro['trans'].std(),size=400000)
            #inverse yeo
            values = pt.inverse_transform(values.reshape(-1, 1)).reshape(1,-1)[0]
            
            array.append(values)
        df = pd.DataFrame(array)
        array=[]
        count1 += 1
        cents = truncate_to_even_cents(df,previous_close)
        df = pd.DataFrame()
        if len(cents.columns) > 0:

            for i in cents.columns:
                #requirements_df[i].mean() <= mean_upper and requirements_df[i].mean() >= mean_lower and
                name = str(i)
                
                plt.plot(cents[i].cumsum(),alpha=.35,label=name+'_Simulation')
                count += 1
            print('count: ',count)
        cents = pd.DataFrame()
    plt.title(symbol+' '+str(dayofweek)+' Simulation with No Constraints')
    plt.savefig('Data/Simulations_Images/'+symbol+'_'+str(dayofweek)+'_General_Only_Day_of_Week_Subset_No_Constraints_SVR-GARCH.png')
    plt.close()

    #return requirements_df 'AMD',
for symbol in ['GOOG']:
    for i in range(0,1,1):
        for j in range(2,3,1):
            create_random_probs(symbol,i,j)
            #general_create_random_probs(symbol,i,j)
