import psycopg2
import psycopg2.extras
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import csv
import requests
from scipy.stats import kstest,normaltest,shapiro
import seaborn as sns
from arch import arch_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.svm import SVR
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PowerTransformer

def get_companies():
    companies = pd.read_csv('companies.csv',header=0)
    return list(companies['Symbol'])

def get_all_holidays_csv():
    holidays = pd.read_csv('all_holidays.csv',header=0)
    holidays = holidays.drop('Unnamed: 0',axis=1)
    holidays['Date'] = pd.to_datetime(holidays['Date']).dt.date
    holi = holidays[holidays['Date'] >= datetime.strptime('2018-01-01','%Y-%m-%d').date()]

    return list(holi['Date'])

def get_all_fed_dates_csv():
    feds = pd.read_csv('all_fed_dates.csv',header=0)
    feds = feds.drop('Unnamed: 0',axis=1)
    feds['Date'] = pd.to_datetime(feds['Date'],format='%Y-%m-%d').dt.date
    feds =feds[feds['Date'] >= datetime.strptime('2018-01-01','%Y-%m-%d').date()]

    return list(feds['Date'])

def get_all_er_dates_csv(symbol):
    er = pd.read_csv('all_er_dates.csv',header=0)
    er = er.drop('Unnamed: 0',axis=1)
    er_dates = er.loc[:,symbol]
    er_dates = pd.to_datetime(er_dates).dt.date
    er_dates = er_dates.dropna(axis=0)

    return list(er_dates)

def get_holidays_csv():
    holidays = pd.read_csv('holidays.csv',header=0)
    holidays['Date'] = pd.to_datetime(holidays['Date'],format='%Y-%m-%d').dt.date
    return holidays

def get_fed_dates_csv():
    feds = pd.read_csv('Fed_Dates.csv',header=0)
    feds['Statement_Release_Date'] = pd.to_datetime(feds['Statement_Release_Date'],format='%Y-%m-%d').dt.date
    return feds

def create_all_good_holidays():
    holidays = get_holidays_csv()
    date_list = []
    for date in holidays['Date']:
        if date.weekday() == 0:
            start_date = date - timedelta(days=2)
            end_date = date + timedelta(2)
            while start_date <= end_date:
            	#account for saturday and sunday when adding and subtracting days.
                if start_date.weekday() != 5 or start_date.weekday() != 6:
                    date_list.append(start_date)
                start_date += timedelta(days=1)
        elif date.weekday() == 1:
            start_date = date - timedelta(days=4)
            end_date = date + timedelta(days=2)
            while start_date <= end_date:
                if start_date.weekday() != 5 or start_date.weekday() != 6:
                    date_list.append(start_date)
                start_date += timedelta(days=1)
        elif date.weekday() == 2:
            start_date = date - timedelta(days=2)
            end_date = date + timedelta(days=2)
            while start_date <= end_date:
                if start_date.weekday() != 5 or start_date.weekday() != 6:
                    date_list.append(start_date)
                start_date += timedelta(days=1)
        elif date.weekday() == 3:
            start_date = date - timedelta(days=2)
            end_date = date + timedelta(days=4)
            while start_date <= end_date:
                if start_date.weekday() != 5 or start_date.weekday() != 6:
                    date_list.append(start_date)
                start_date += timedelta(days=1)
        elif date.weekday() == 4:
            start_date = date - timedelta(days=2)
            end_date = date + timedelta(days=4)
            while start_date <= end_date:
                if start_date.weekday() != 5 or start_date.weekday() != 6:
                    date_list.append(start_date)
                start_date += timedelta(days=1)
    df = pd.DataFrame(date_list,columns=['Date'])
    df.to_csv('all_holidays.csv',header=True)
    

def create_all_good_fed_dates():
    fed_dates = get_fed_dates_csv()
    date_list = []
    for date in fed_dates['Statement_Release_Date']:
        if date.weekday() == 0:
            start_date = date - timedelta(days=4)
            end_date = date + timedelta(1)
            while start_date <= end_date:
                if start_date.weekday() != 5 or start_date.weekday() != 6:
                    date_list.append(start_date)
                start_date += timedelta(days=1)
        elif date.weekday() == 1:
            start_date = date - timedelta(days=1)
            end_date = date + timedelta(days=1)
            while start_date <= end_date:
                if start_date.weekday() != 5 or start_date.weekday() != 6:
                    date_list.append(start_date)
                start_date += timedelta(days=1)
        elif date.weekday() == 2:
            start_date = date - timedelta(days=1)
            end_date = date + timedelta(days=1)
            while start_date <= end_date:
                if start_date.weekday() != 5 or start_date.weekday() != 6:
                    date_list.append(start_date)
                start_date += timedelta(days=1)
        elif date.weekday() == 3:
            start_date = date - timedelta(days=1)
            end_date = date + timedelta(days=1)
            while start_date <= end_date:
                if start_date.weekday() != 5 or start_date.weekday() != 6:
                    date_list.append(start_date)
                start_date += timedelta(days=1)
        elif date.weekday() == 4:
            start_date = date - timedelta(days=1)
            end_date = date + timedelta(days=4)
            while start_date <= end_date:
                if start_date.weekday() != 5 or start_date.weekday() != 6:
                    date_list.append(start_date)
                start_date += timedelta(days=1)
    df = pd.DataFrame(date_list,columns=['Date'])
    df.to_csv('all_fed_dates.csv',header=True)    
    return date_list

def create_all_good_earnings_dates():
    all_er_dates = pd.DataFrame()
    er_dates = pd.read_csv('filing_dates.csv',header=0)
    er_dates = er_dates.drop('Unnamed: 0',axis=1)
    symbols = pd.read_csv('companies.csv',header=0)
    for symbol in list(symbols['Symbol']):
        alle_dates = {}
        er_dates[symbol.upper()] = pd.to_datetime(er_dates[symbol.upper()]).dt.date
        date_list = []
        for date in list(er_dates[symbol.upper()]):
            
            start_date = date - timedelta(days=7)
            end_date = date + timedelta(7)
            while start_date <= end_date:
                
                if (start_date.weekday() != 5) and (start_date.weekday() != 6):
                    
                    date_list.append(start_date)
                start_date += timedelta(days=1)
        alle_dates[symbol.upper()] = date_list
        df = pd.DataFrame.from_dict(alle_dates)

        all_er_dates = pd.concat([all_er_dates,df],axis=1)
    all_er_dates.to_csv('all_er_dates.csv',header=True) 


def create_full_data(symbols):
    for symbol in symbols:
        con = psycopg2.connect(host=os.environ['RT_DATABASE_HOST'],
                               database=os.environ['RT_DATABASE_NAME'],
                               user=os.environ['RT_DATABASE_USER'],
                               password=os.environ['RT_DATABASE_PASS'])
        cur = con.cursor()
        
        sql = "Select timestamp, close From intraday_prices where symbol=%s and timestamp::time >='09:30:00' and timestamp::time < '16:00:00' and timestamp::date >='2018-10-29' and timestamp::date <='2023-10-26'  order by timestamp"
        cur.execute(sql,(symbol,))
        data = pd.DataFrame(cur.fetchall(),columns=['timestamp','close'])
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['change'] = data['close'].pct_change()
        data = data.dropna(axis=0)
        data = data.drop('close',axis=1)
        cur.close()
        con.close()
 
        data.to_csv('Data/Daily_Intraday/'+symbol.upper()+'_intraday.csv',header=True)
def create_open_data(symbols):
    for symbol in symbols:
        csv = pd.read_csv('Data/Daily_Intraday/'+symbol.upper()+'_intraday.csv',header=0)
        csv['timestamp'] = pd.to_datetime(csv['timestamp'])
        csv['time'] = csv['timestamp'].dt.time
        open_data = csv[csv['time'] == datetime.strptime('09:30:00','%H:%M:%S').time()].reset_index()
        open_data = open_data.drop(['index','Unnamed: 0','time'],axis=1)
        open_data.to_csv('Data/Daily_Intraday/Open_Changes/'+symbol.upper()+'_open_change.csv')

def create_individual_intraday(symbols):
    for symbol in symbols:
        csv = pd.read_csv('Data/Full_Data/'+symbol.lower()+'_full_data.csv',header=0)
        csv['timestamp'] = pd.to_datetime(csv['timestamp'])
        temp = csv[csv['symbol'] == symbol].reset_index()        
        temp['change'] = temp['close'].pct_change()
        temp = temp.drop(['index','symbol','open','high','low','close'],axis=1)
        
        temp.to_csv('Data/Daily_Intraday/'+symbol+'_intraday.csv')
def create_close_over_close(symbols):
    
    for symbol in symbols:
        df = pd.DataFrame()
        csv = pd.read_csv('Data/Daily_Intraday/'+symbol.upper()+'_intraday.csv',header=0)
        csv['timestamp'] = pd.to_datetime(csv['timestamp'])
        dates = list(csv['timestamp'].dt.date.unique())
        times = list(csv['timestamp'].dt.time.unique())
        for date in dates:
            pc = dates.index(date)
            if pc != 0:
                prev = csv[csv['timestamp'].dt.date == date].reset_index()
                prev['cumsum'] = prev['change'].cumsum()
                prev = prev.drop(['index','Unnamed: 0','change'],axis=1)
                prev = prev[prev['timestamp'].dt.time == times[-2]].reset_index()
                prev = prev.drop('index',axis=1)
                df = pd.concat([df,prev],axis=0)

        df.to_csv('Data/Close_over_Close/'+symbol.upper()+'_close_change.csv')
        
def intraday_descriptive_stats(symbols):

    for symbol in symbols:
        print(symbol)
        csv = pd.read_csv('Data/Daily_Intraday/'+symbol.upper()+'_intraday.csv',header=0)
        csv['timestamp'] = pd.to_datetime(csv['timestamp'])
        days = csv['timestamp'].dt.date.unique()
        csv=csv.drop('Unnamed: 0',axis=1)
        ms = {'date':[],'means':[],'std':[]}
        for day in days:
            temp = csv[csv['timestamp'].dt.date == day].reset_index()
            temp = temp.drop('index',axis=1)
            ms['date'].append(day)
            ms['means'].append(temp['change'].mean())
            ms['std'].append(temp['change'].std())
        df = pd.DataFrame.from_dict(ms)
        df.to_csv('Data/Daily_Intraday/Descriptive_Stats/'+symbol.upper()+'_description.csv',header=True)
        
def intraday_descriptive_mean_std_norms(symbols):
    descr = {'symbol':[],'dow':[],'mean_normal':[],'std_normal':[],'transformed':[]}
    for symbol in symbols:
        print(symbol)
        csv = pd.read_csv('Data/Daily_Intraday/Descriptive_Stats/'+symbol.upper()+'_description.csv',header=0)
        csv['date'] = pd.to_datetime(csv['date'])
        csv=csv.drop('Unnamed: 0',axis=1)

        for i in range(0,5,1):
            temp = csv[csv['date'].dt.dayofweek == i].reset_index()
            temp = temp.drop(['index'],axis=1)
            pt = PowerTransformer(method='yeo-johnson',standardize=True)
            temp['mean_trans'] = pt.fit_transform(np.array(temp['means']).reshape(len(temp['means']),1)).reshape(1,-1)[0]
            temp['std_trans'] = pt.fit_transform(np.array(temp['std']).reshape(len(temp['std']),1)).reshape(1,-1)[0]
            
            ks = kstest(temp['mean_trans'],'norm')
            descr['symbol'].append(symbol)
            descr['dow'].append(i)
            descr['transformed'].append(True)
            if ks.pvalue > .05:
                descr['mean_normal'].append(True)
            else:
                descr['mean_normal'].append(False)
                
            nks = kstest(temp['std_trans'],'norm')
            if nks.pvalue > .05:
                descr['std_normal'].append(True)
            else:
                descr['std_normal'].append(False)
    df = pd.DataFrame.from_dict(descr)
    df.to_csv('Data/Daily_Intraday/Descriptive_Stats/description_mean_std_normal.csv',header=True)                            

def intraday_descriptive_mean_std_no_p_norms(symbols):
    descr = {'symbol':[],'dow':[],'mean_normal':[],'std_normal':[],'transformed':[]}
    for symbol in symbols:
        print(symbol)
        csv = pd.read_csv('Data/Daily_Intraday/Descriptive_Stats/'+symbol.upper()+'_description.csv',header=0)
        csv['date'] = pd.to_datetime(csv['date'])
        csv=csv.drop('Unnamed: 0',axis=1)
        remove_h = get_all_holidays_csv()
        remove_f = get_all_fed_dates_csv()
        remove_e = get_all_er_dates_csv(symbol)
        no_phenom = list(set(remove_h+remove_f+remove_e))
        temp =csv[~csv['date'].isin(list(no_phenom))]
        temp = temp.reset_index()
        temp = temp.drop('index',axis=1)

        for i in range(0,5,1):
            temp1 = temp[temp['date'].dt.dayofweek == i].reset_index()
            temp1 = temp1.drop(['index'],axis=1)
            pt = PowerTransformer(method='yeo-johnson',standardize=True)
            temp1['mean_trans'] = pt.fit_transform(np.array(temp1['means']).reshape(len(temp1['means']),1)).reshape(1,-1)[0]
            temp1['std_trans'] = pt.fit_transform(np.array(temp1['std']).reshape(len(temp1['std']),1)).reshape(1,-1)[0]
            
            ks = kstest(temp1['mean_trans'],'norm')
            descr['symbol'].append(symbol)
            descr['dow'].append(i)
            descr['transformed'].append(True)
            if ks.pvalue > .05:
                descr['mean_normal'].append(True)
            else:
                descr['mean_normal'].append(False)
                
            nks = kstest(temp1['std_trans'],'norm')
            if nks.pvalue > .05:
                descr['std_normal'].append(True)
            else:
                descr['std_normal'].append(False)
    df = pd.DataFrame.from_dict(descr)
    df.to_csv('Data/Daily_Intraday/Descriptive_Stats/description_mean_std_normal_no_p.csv',header=True) 
        
def fill_missing_times(symbols):

    good_times = np.array(pd.date_range("09:30:00",'15:59:00', freq = "1min").time)
    
    for symbol in symbols:
        print(symbol)
        csv = pd.read_csv('Data/Daily_Intraday/'+symbol.upper()+'_intraday.csv',header=0)
        csv['timestamp'] = pd.to_datetime(csv['timestamp'])
        missing_time = {'timestamp':[],'change':[]}
        dates = csv['timestamp'].dt.date.unique()
        
        for date in dates:

            temp = csv[csv['timestamp'].dt.date == date]
            temp = temp.drop(['Unnamed: 0'],axis=1)
            temp_times = np.array(temp['timestamp'].dt.time.unique())
            if (210 <= len(temp_times) <= 215) and (len(temp_times) != 390):
            
                main_list = set(good_times)^set(temp_times)
                for i in main_list:
                    
                    fresh = datetime.combine(date, i)
                    missing_time['timestamp'].append(fresh)
                    missing_time['change'].append(0)

        df = pd.DataFrame.from_dict(missing_time)
        good_df = pd.concat([csv,df])
        good_df = good_df.sort_values('timestamp')
        good_df = good_df.drop('Unnamed: 0',axis=1)
        print(good_df.head())
        good_df.to_csv('Data/Daily_Intraday/'+symbol.upper()+'_intraday.csv',header=True)
        
def minute_to_minute_og_norm_check(symbols):
        norm_results = {'symbol':[],'normal_minutes':[],'normalized':[],'transformed':[]}
        for symbol in symbols:
            nm = 0
            normed = 0
            transf = 0
            df = pd.read_csv('Data/Daily_Intraday/'+symbol.upper()+'_intraday.csv',header=0)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            times = df['timestamp'].dt.time.unique()
            for time in times:
                temp = df[df['timestamp'].dt.time == time].reset_index()
                temp['normed'] = (temp['change']-temp['change'].mean())/temp['change'].std()
                pt = PowerTransformer(method='yeo-johnson',standardize=True)
                temp['trans'] = pt.fit_transform(np.array(temp['change']).reshape(len(temp['change']),1)).reshape(1,-1)[0]
                temp = temp.drop(['index','Unnamed: 0'],axis=1)
                ks = kstest(temp['change'],'norm')
                if ks.pvalue > .05:
                    nm += 1
                nks = kstest(temp['normed'],'norm')
                if nks.pvalue > .05:
                    normed += 1
                tks = kstest(temp['trans'],'norm')
                if tks.pvalue > .05:
                    transf += 1
            norm_results['symbol'].append(symbol)
            norm_results['normal_minutes'].append(nm)
            norm_results['normalized'].append(normed)
            norm_results['transformed'].append(transf)
        ndf = pd.DataFrame.from_dict(norm_results)
        
        ndf.to_csv('Data/Daily_Intraday/Descriptive_Stats/og_norm_counts.csv',header=True)
    
def minute_to_minute_no_phenomena_norm_check(symbols):
        norm_results = {'symbol':[],'dayofweek':[],'group':[],'normal_minutes':[],'normalized':[],'transformed':[]}
        for symbol in symbols:
            df = pd.read_csv('Data/Daily_Intraday/'+symbol.upper()+'_intraday.csv',header=0)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            times = df['timestamp'].dt.time.unique()
            for i in range(0,5,1):
                nm = 0
                normed = 0
                transf = 0
                for time in times:
                    temp = df[(df['timestamp'].dt.time == time) & (df['timestamp'].dt.dayofweek==i)].reset_index()
                    temp['Date'] = temp['timestamp'].dt.date
                    temp = temp.drop(['index','Unnamed: 0'],axis=1)
                    
                    remove_h = get_all_holidays_csv()
                    remove_f = get_all_fed_dates_csv()
                    remove_e = get_all_er_dates_csv(symbol)
                    no_phenom = list(set(remove_h+remove_f+remove_e))
                    pt = PowerTransformer(method='yeo-johnson',standardize=True)
                    no_p =temp[~temp['Date'].isin(list(no_phenom))]
                    no_p = no_p.reset_index()
                    no_p = no_p.drop('index',axis=1)
                    no_p['normed'] = (no_p['change']-no_p['change'].mean())/no_p['change'].std()
                    no_p['trans'] = pt.fit_transform(np.array(no_p['change']).reshape(len(no_p['change']),1)).reshape(1,-1)[0]

                             
                    ks = kstest(no_p['change'],'norm')
                    if ks.pvalue > .05:
                        nm += 1
                    nks = kstest(no_p['normed'],'norm')
                    if nks.pvalue > .05:
                        normed += 1
                    tks = kstest(no_p['trans'],'norm')
                    if tks.pvalue > .05:
                        transf += 1
                norm_results['symbol'].append(symbol)
                norm_results['dayofweek'].append(i)
                norm_results['group'].append('no_phenom')
                norm_results['normal_minutes'].append(nm)
                norm_results['normalized'].append(normed)
                norm_results['transformed'].append(transf)
        ndf = pd.DataFrame.from_dict(norm_results)
        ndf = ndf.sort_values(by=['dayofweek','symbol'])
        print(ndf.head(13))
        ndf.to_csv('Data/Daily_Intraday/Descriptive_Stats/no_phenom_norm_counts.csv',header=True)

def dow_minute_means_only(symbols):
    norm_results = {'symbol':[],'dayofweek':[],'group':[],'normalized':[],'transformed':[]}
    for symbol in symbols:
        df = pd.read_csv('Data/Daily_Intraday/'+symbol.upper()+'_intraday.csv',header=0)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        times = df['timestamp'].dt.time.unique()
        for i in range(0,5,1):
            normed = 0
            transf=0
            for time in times:

                temp = df[(df['timestamp'].dt.time == time) & (df['timestamp'].dt.dayofweek==i)].reset_index()
                temp['Date'] = temp['timestamp'].dt.date
                temp['normed'] = (temp['change']-temp['change'].mean())/temp['change'].std()
                pt = PowerTransformer(method='yeo-johnson',standardize=True)
                temp['trans'] = pt.fit_transform(np.array(temp['change']).reshape(len(temp['change']),1)).reshape(1,-1)[0]
                temp = temp.drop(['index','Unnamed: 0'],axis=1)

                temp['normed'] = (temp['change']-temp['change'].mean())/temp['change'].std()
                temp['trans'] = pt.fit_transform(np.array(temp['change']).reshape(len(temp['change']),1)).reshape(1,-1)[0]
                ks = kstest(temp['normed'],'norm')
                if ks.pvalue > .05:
                    normed += 1

                tks = kstest(temp['trans'],'norm')
                if tks.pvalue > .05:
                    transf += 1
            norm_results['symbol'].append(symbol)
            norm_results['dayofweek'].append(i)
            norm_results['group'].append('dow only')
            norm_results['normalized'].append(normed)
            norm_results['transformed'].append(transf)
            print(symbol,' ',i,' ',normed,' ',transf)
    ndf = pd.DataFrame.from_dict(norm_results)
    ndf = ndf.sort_values(by=['dayofweek','symbol'])
    print(ndf.head(13))
    ndf.to_csv('Data/Daily_Intraday/Descriptive_Stats/daily_only_norm_counts.csv',header=True)

def daily_minute_means_csv(symbols):
    
    for symbol in symbols:
        probs = {}
        df = pd.read_csv('Data/Daily_Intraday/'+symbol.upper()+'_intraday.csv',header=0)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        times = df['timestamp'].dt.time.unique()
        df['change'] = df['change']*10**3

        df = df.dropna(axis=0)

        for i in range(0,5,1):
            
            start = datetime.strptime('09:30','%H:%M')
            end = datetime.strptime('16:00','%H:%M')
            
            for time in times:
                
                temp = df[(df['timestamp'].dt.time == time) & (df['timestamp'].dt.dayofweek == i)].reset_index()
                temp = temp.drop(['Unnamed: 0','index'],axis=1)

                temp['close_normed'] = ((temp['change']-temp['change'].mean())/temp['change'].std())

                pt = PowerTransformer(method='yeo-johnson',standardize=True)
                temp['trans'] = pt.fit_transform(np.array(temp['change']).reshape(len(temp['change']),1)).reshape(1,-1)[0]
                print(temp['trans'].mean(),' ',temp['trans'].std())
                start += timedelta(minutes=1)
    cursor.close()
    conn.close()

def populate_daily_minute_means_table():
    daily_means = pd.DataFrame()
    
    for symbol in ['AMD','GOOG']:  
        df = pd.read_csv('Data/Daily_Intraday/'+symbol.upper()+'_intraday.csv',header=0)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        

        df = df.dropna(axis=0)

        for i in range(0,5,1):
            daily_minute_means = {'timestamp':[],'day_of_week':[],'change_mean':[],'change_std':[],'normed':[],'normed_mean':[],'normed_std':[],'trans_mean':[],'trans_std':[]}
            start = datetime.strptime('09:30','%H:%M')
            end = datetime.strptime('16:00','%H:%M')
            
            while start.time() < end.time():
                temp = df[(df['timestamp'].dt.time == start.time()) & (df['timestamp'].dt.dayofweek==i)].reset_index()
                temp['Date'] = temp['timestamp'].dt.date
                temp = temp.drop(['index','Unnamed: 0'],axis=1)
                remove_h = get_all_holidays_csv()
                remove_f = get_all_fed_dates_csv()
                remove_e = get_all_er_dates_csv(symbol)
                no_phenom = list(set(remove_h+remove_f+remove_e))
                
                no_p = temp[~temp['Date'].isin(no_phenom)].reset_index()
                no_p = no_p.drop('index',axis=1)
                pt = PowerTransformer(method='yeo-johnson',standardize=True)
                no_p['trans'] = pt.fit_transform(np.array(no_p['change']).reshape(len(no_p['change']),1)).reshape(1,-1)[0]
                #print(no_p['trans'].mean(),' ',no_p['trans'].std())
                no_p['normed'] = ((no_p['change']-no_p['change'].mean())/no_p['change'].std())
                daily_minute_means['timestamp'].append(start.time())
                daily_minute_means['day_of_week'].append(i)
                daily_minute_means['change_mean'].append(no_p['change'].mean())
                daily_minute_means['change_std'].append(no_p['change'].std())
                ks = kstest(no_p['normed'],'norm')
                if ks.pvalue > .05:
                    daily_minute_means['normed'].append(True)
                else:
                    daily_minute_means['normed'].append(False)
                daily_minute_means['normed'].append(True)
                daily_minute_means['normed_mean'].append(no_p['normed'].mean())
                daily_minute_means['normed_std'].append(no_p['normed'].std())
                daily_minute_means['trans_mean'].append(no_p['normed'].mean())
                daily_minute_means['trans_std'].append(no_p['normed'].std())
                start += timedelta(minutes=1)
            dmm = pd.DataFrame.from_dict(daily_minute_means)
            dmm.to_csv('Data/Daily_Intraday/Daily_Means/'+symbol+'_'+str(i)+'_daily_minute_means.csv',header=True)
            print(dmm.head())
def open_norms(symbols):
    descr = {'symbol':[],'dow':[],'normal':[],'transformed':[]}
    for symbol in symbols:
        print(symbol)
        csv = pd.read_csv('Data/Daily_Intraday/Open_Changes/'+symbol.upper()+'_open_change.csv',header=0)
        csv['timestamp'] = pd.to_datetime(csv['timestamp'])
        csv['date'] = csv['timestamp'].dt.date
        csv['date'] = pd.to_datetime(csv['date'])
        csv=csv.drop(['Unnamed: 0','timestamp'],axis=1)

        for i in range(0,5,1):
            temp = csv[csv['date'].dt.dayofweek == i].reset_index()
            temp = temp.drop(['index'],axis=1)
            pt = PowerTransformer(method='yeo-johnson',standardize=True)
            temp['trans'] = pt.fit_transform(np.array(temp['change']).reshape(len(temp['change']),1)).reshape(1,-1)[0]
            
            ks = kstest(temp['trans'],'norm')
            descr['symbol'].append(symbol)
            descr['dow'].append(i)
            descr['transformed'].append(True)
            if ks.pvalue > .05:
                descr['normal'].append(True)
            else:
                descr['normal'].append(False)

    df = pd.DataFrame.from_dict(descr)
    df.to_csv('Data/Daily_Intraday/Descriptive_Stats/description_open_normal.csv',header=True)                            

def open_no_p_norms(symbols):
    descr = {'symbol':[],'dow':[],'normal':[],'transformed':[]}
    for symbol in symbols:
        print(symbol)
        csv = pd.read_csv('Data/Daily_Intraday/Open_Changes/'+symbol.upper()+'_open_change.csv',header=0)
        csv['timestamp'] = pd.to_datetime(csv['timestamp'])
        csv['date'] = csv['timestamp'].dt.date
        csv['date'] = pd.to_datetime(csv['date'])
        csv=csv.drop(['Unnamed: 0','timestamp'],axis=1)
        remove_h = get_all_holidays_csv()
        remove_f = get_all_fed_dates_csv()
        remove_e = get_all_er_dates_csv(symbol)
        no_phenom = list(set(remove_h+remove_f+remove_e))
        temp =csv[~csv['date'].isin(list(no_phenom))]
        temp = temp.reset_index()
        temp = temp.drop('index',axis=1)

        for i in range(0,5,1):
            temp1 = temp[temp['date'].dt.dayofweek == i].reset_index()
            temp1 = temp1.drop(['index'],axis=1)
            pt = PowerTransformer(method='yeo-johnson',standardize=True)
            temp['trans'] = pt.fit_transform(np.array(temp['change']).reshape(len(temp['change']),1)).reshape(1,-1)[0]
            
            ks = kstest(temp['trans'],'norm')
            descr['symbol'].append(symbol)
            descr['dow'].append(i)
            descr['transformed'].append(True)
            if ks.pvalue > .05:
                descr['normal'].append(True)
            else:
                descr['normal'].append(False)
    df = pd.DataFrame.from_dict(descr)
    df.to_csv('Data/Daily_Intraday/Descriptive_Stats/description_open_normal_no_p.csv',header=True)

def close_norms(symbols):
    descr = {'symbol':[],'dow':[],'normal':[],'transformed':[]}
    for symbol in symbols:
        print(symbol)
        csv = pd.read_csv('Data/Close_over_Close/'+symbol.upper()+'_close_change.csv',header=0)
        csv['timestamp'] = pd.to_datetime(csv['timestamp'])
        csv['date'] = csv['timestamp'].dt.date
        csv['date'] = pd.to_datetime(csv['date'])
        csv=csv.drop(['Unnamed: 0','timestamp'],axis=1)

        for i in range(0,5,1):
            temp = csv[csv['date'].dt.dayofweek == i].reset_index()
            temp = temp.drop(['index'],axis=1)
            pt = PowerTransformer(method='yeo-johnson',standardize=True)
            temp['trans'] = pt.fit_transform(np.array(temp['cumsum']).reshape(len(temp['cumsum']),1)).reshape(1,-1)[0]
            
            ks = kstest(temp['trans'],'norm')
            descr['symbol'].append(symbol)
            descr['dow'].append(i)
            descr['transformed'].append(True)
            if ks.pvalue > .05:
                descr['normal'].append(True)
            else:
                descr['normal'].append(False)

    df = pd.DataFrame.from_dict(descr)
    df.to_csv('Data/Daily_Intraday/Descriptive_Stats/description_close_normal.csv',header=True)                            

def close_no_p_norms(symbols):
    descr = {'symbol':[],'dow':[],'normal':[],'transformed':[]}
    for symbol in symbols:
        print(symbol)
        csv = pd.read_csv('Data/Close_over_Close/'+symbol.upper()+'_close_change.csv',header=0)
        csv['timestamp'] = pd.to_datetime(csv['timestamp'])
        csv['date'] = csv['timestamp'].dt.date
        csv['date'] = pd.to_datetime(csv['date'])
        csv=csv.drop(['Unnamed: 0','timestamp'],axis=1)
        remove_h = get_all_holidays_csv()
        remove_f = get_all_fed_dates_csv()
        remove_e = get_all_er_dates_csv(symbol)
        no_phenom = list(set(remove_h+remove_f+remove_e))
        temp =csv[~csv['date'].isin(list(no_phenom))]
        temp = temp.reset_index()
        temp = temp.drop('index',axis=1)

        for i in range(0,5,1):
            temp1 = temp[temp['date'].dt.dayofweek == i].reset_index()
            temp1 = temp1.drop(['index'],axis=1)
            pt = PowerTransformer(method='yeo-johnson',standardize=True)
            temp['trans'] = pt.fit_transform(np.array(temp['cumsum']).reshape(len(temp['cumsum']),1)).reshape(1,-1)[0]
            
            ks = kstest(temp['trans'],'norm')
            descr['symbol'].append(symbol)
            descr['dow'].append(i)
            descr['transformed'].append(True)
            if ks.pvalue > .05:
                descr['normal'].append(True)
            else:
                descr['normal'].append(False)
    df = pd.DataFrame.from_dict(descr)
    df.to_csv('Data/Daily_Intraday/Descriptive_Stats/description_close_normal_no_p.csv',header=True)
    
symbols = get_companies()
##
##create_full_data(symbols)
#create_open_data(symbols)
#create_close_over_close(symbols)
##create_individual_intraday(symbols)
##intraday_descriptive_stats(symbols)
#daily_ohlc_stats(symbols)
##intraday_daily_mm_mean_table(symbols)
##intraday_daily_mm_std_table(symbols)
##create_all_good_holidays()
##create_all_good_fed_dates()
##create_all_good_earnings_dates()
#fill_missing_times(symbols)
##minute_to_minute_og_norm_check(symbols)
##minute_to_minute_no_phenomena_norm_check(symbols)
##dow_minute_means_only(symbols)
##intraday_descriptive_mean_std_norms(symbols)
##intraday_descriptive_mean_std_no_p_norms(symbols)
##open_norms(['AMD','GOOG'])
##open_no_p_norms(['AMD','GOOG'])
##close_norms(['AMD','GOOG'])
##close_no_p_norms(['AMD','GOOG'])
#populate_daily_minute_means_table()
##create_all_good_earnings_dates()
##intraday_descriptive_mean_std_norms(['AMD','GOOG'])
##intraday_descriptive_mean_std_no_p_norms(['AMD','GOOG'])
