import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import csv
import seaborn as sns
from arch import arch_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PowerTransformer
from data_generation import get_all_holidays_csv,get_all_fed_dates_csv,get_all_er_dates_csv
from scipy.stats import kstest,norm
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def date_removal_close_over_close(symbol,dayofweek):

    daily = pd.read_csv('Data/Close_over_Close/close_over_close.csv',header=0)
    daily['timestamp'] = pd.to_datetime(daily['timestamp'])
    daily = daily[(daily['symbol'] == symbol) & (daily['timestamp'].dt.dayofweek==dayofweek)].reset_index()
    remove_h = get_all_holidays_csv()
    remove_f = get_all_fed_dates_csv()
    remove_e = get_all_er_dates_csv(symbol)
    no_phenom = list(set(remove_h+remove_f+remove_e))

    no_p =  daily[~daily['timestamp'].isin(no_phenom)].reset_index()
    no_p = no_p.drop(['level_0','Unnamed: 0','index'],axis=1)
    
    return no_p

def date_removal_intraday(symbol,dayofweek):

    daily = pd.read_csv('Data/Daily_Intraday/Descriptive_Stats/'+symbol+'_description.csv',header=0)
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily[daily['date'].dt.dayofweek==dayofweek].reset_index()

    remove_h = get_all_holidays_csv()
    remove_f = get_all_fed_dates_csv()
    remove_e = get_all_er_dates_csv(symbol)
    no_phenom = list(set(remove_h+remove_f+remove_e))

    no_p =  daily[~daily['date'].isin(no_phenom)].reset_index()
    no_p = no_p.drop(['level_0','Unnamed: 0','index'],axis=1)

    return no_p

def date_removal_open(symbol,dayofweek):

    daily = pd.read_csv('Data/Daily_Intraday/'+symbol+'_intraday.csv',header=0)
    daily['timestamp'] = pd.to_datetime(daily['timestamp'])
    
    daily = daily[(daily['timestamp'].dt.time == datetime.strptime('09:30:00','%H:%M:%S').time())].reset_index()
    daily['date'] = daily['timestamp'].dt.date
    daily=daily.drop(['Unnamed: 0','index'],axis=1)
    daily = daily[daily['timestamp'].dt.dayofweek==dayofweek].reset_index()
    remove_h = get_all_holidays_csv()
    remove_f = get_all_fed_dates_csv()
    remove_e = get_all_er_dates_csv(symbol)
    no_phenom = list(set(remove_h+remove_f+remove_e))

    no_p =  daily[~daily['date'].isin(no_phenom)].reset_index()
    no_p = no_p.drop(['level_0','index'],axis=1)
    
    return no_p

# transform a time series dataset into a supervised learning dataset
def series_to_timesequence(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values
 
### split a univariate dataset into train/test sets
##def train_test_split(data, n_test):
##    return data[:-n_test, :], data[-n_test:, :]
 
### fit an xgboost model and make a one step prediction
##def xgboost_forecast(train, testX):
##    # transform list into array
##    train = np.asarray(train)
##    # split into input and output columns
##    trainX, trainy = train[:, :-1], train[:, -1]
##    # fit model
##    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
##    model.fit(trainX, trainy)
##    # make a one-step prediction
##    yhat = model.predict(np.asarray([testX]))
##    return yhat[0]
  
def moving_average_inverse(last_ma_values,last_predicted_value,divide_by):
    
    one = (last_predicted_value)*(len(last_ma_values)+1)
    two = sum(last_ma_values)
    three = (one-two)/divide_by

    return three

# split a univariate sequence
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
    # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
    X.append(seq_x)
    y.append(seq_y)
    return np.array(X), np.array(y)
 
def individual_garch_intraday_open(symbol, dayofweek,xgb_pred,ci):
    results_dict = {'symbol':[],'Model':[],'Date':[],'dayofweek':[],'RMSE':[],'Last_Prediction':[],'Inv_Ma':[],'Errors_ND':[],'Last_Error':[],'Error_Mean':[],'Error_Std':[],'Upper_Bound':[],'Lower_Bound':[]}
    ## Data WITHOUT Holidays,Fed Dates, Earnings Dates
    no_phenomena = date_removal_open(symbol,dayofweek).reset_index().copy()
    no_phenomena['change'].loc[-1] = xgb_pred
    n_returns = no_phenomena['change']*1000
    n_realized_vol = n_returns.rolling(6).mean()
    n_n = int(round(len(no_phenomena['change'])*.3,0))
    n_split_date = n_returns.iloc[-n_n:].index
    ## SVR-Garch Linear No Phenomena
    n_realized_vol = pd.DataFrame(n_realized_vol)
    n_realized_vol.reset_index(drop=True, inplace=True)
    n_returns_svm = n_returns
    n_returns_svm = n_returns_svm.reset_index()
    
    del n_returns_svm['index']
    n_X = pd.concat([n_realized_vol, n_returns_svm], axis=1, ignore_index=True)
    n_X = n_X[5:].copy()
    n_X = n_X.reset_index()
    n_X.drop('index', axis=1, inplace=True)
    n_realized_vol = n_realized_vol.dropna().reset_index()
    n_realized_vol.drop('index', axis=1, inplace=True)
    svr_lin = SVR(kernel='linear')
    para_grid = {'gamma': sp_rand(),'C': sp_rand(),'epsilon': sp_rand()} 
    clf = RandomizedSearchCV(svr_lin, para_grid)
    clf.fit(n_X.iloc[:-n_n].values,n_realized_vol.iloc[1:-(n_n-1)].values.reshape(-1,))
    
    n_predict_svr_lin = clf.predict(n_X.iloc[-n_n:])
    n_predict_svr_lin = pd.DataFrame(n_predict_svr_lin)
    n_predict_svr_lin.index = n_returns.iloc[-n_n:].index
    n_rmse_svr = np.sqrt(mse(n_realized_vol.iloc[-n_n:],n_predict_svr_lin))
    errors = ((n_realized_vol['change'].iloc[-n_n:].reset_index()['change']) -(n_predict_svr_lin[0].reset_index()[0]))
    errors_n = ((errors - errors.mean())/errors.std())
    eks = kstest(list(errors_n),'norm')
    enormal = False
    if eks.pvalue >= .05:
        enormal=True

    ma = moving_average_inverse(n_X.iloc[-6:-1][1],n_predict_svr_lin[0].iloc[-1],1)/1000
    ub =  n_predict_svr_lin[0].iloc[-1] + (errors.mean() + errors.std()*ci)
    lb = n_predict_svr_lin[0].iloc[-1] + (errors.mean()-errors.std()*ci)
    upper = moving_average_inverse(n_X.iloc[-6:-1][1],ub,1)/1000
    lower = moving_average_inverse(n_X.iloc[-6:-1][1],lb,1)/1000
    if upper > lower:
        results_dict['Upper_Bound'].append(upper)
        results_dict['Lower_Bound'].append(lower)
    else:
        results_dict['Upper_Bound'].append(lower)
        results_dict['Lower_Bound'].append(upper)
    results_dict['symbol'].append(symbol)
    results_dict['Model'].append('SVR_Linear_Garch')
    results_dict['Date'].append(no_phenomena['timestamp'].iloc[-1])
    results_dict['dayofweek'].append(dayofweek)
    results_dict['RMSE'].append(round(n_rmse_svr,6))
    results_dict['Last_Prediction'].append(n_predict_svr_lin[0].iloc[-1]/1000)
    results_dict['Inv_Ma'].append(ma)
    results_dict['Errors_ND'].append(enormal)
    results_dict['Last_Error'].append(errors.iloc[-1]/1000)
    results_dict['Error_Mean'].append(errors.mean()/1000)
    results_dict['Error_Std'].append(errors.std()/1000)

    n_realized_vol.index = n_returns.iloc[5:].index

##    plt.figure(figsize=(10, 6))
##    plt.plot(n_realized_vol / 1000, label='Real Openings No Phenomena')
##    plt.plot(n_predict_svr_lin / 1000, label='Opening Prediction-SVR-GARCH No Phenomena')
##    plt.title(symbol+' '+str(dayofweek)+'_'+str(ci)+'stds No Phenomena Openings Prediction with SVR-GARCH (Linear)', fontsize=12)
##    plt.legend()
##    plt.savefig('Data/Garch_Results/Images/'+symbol+'_'+str(dayofweek)+'_'+str(ci)+'stds_No_Phenomena_Open_SVR-GARCH.png')
##    plt.close()
##
##    df= pd.DataFrame.from_dict(results_dict)
##    df.to_csv('Data/Garch_Results/'+symbol+'_'+str(dayofweek)+'_'+str(ci)+'stds_Intraday_Open_Garch_Results.csv',header=True)

def individual_garch_intraday_std(symbol, dayofweek,xgb_pred,ci):
    results_dict = {'symbol':[],'Model':[],'Date':[],'dayofweek':[],'RMSE':[],'Last_Prediction':[],'Inv_Ma':[],'Errors_ND':[],'Last_Error':[],'Error_Mean':[],'Error_Std':[],'Upper_Bound':[],'Lower_Bound':[]}
    ## Data WITHOUT Holidays,Fed Dates, Earnings Dates
    no_phenomena = date_removal_intraday(symbol,dayofweek).reset_index().copy()
    no_phenomena['std'].loc[-1] = xgb_pred
    n_returns = no_phenomena['std']*10000
    n_realized_vol = n_returns.rolling(6).mean()
    n_n = int(round(len(no_phenomena['std'])*.3,0))
    n_split_date = n_returns.iloc[-n_n:].index
    ## SVR-Garch Linear No Phenomena
    n_realized_vol = pd.DataFrame(n_realized_vol)
    n_realized_vol.reset_index(drop=True, inplace=True)
    n_returns_svm = n_returns
    n_returns_svm = n_returns_svm.reset_index()
    
    del n_returns_svm['index']
    n_X = pd.concat([n_realized_vol, n_returns_svm], axis=1, ignore_index=True)
    n_X = n_X[5:].copy()
    n_X = n_X.reset_index()
    n_X.drop('index', axis=1, inplace=True)
    n_realized_vol = n_realized_vol.dropna().reset_index()
    n_realized_vol.drop('index', axis=1, inplace=True)
    svr_lin = SVR(kernel='linear')
    para_grid = {'gamma': sp_rand(),'C': sp_rand(),'epsilon': sp_rand()} 
    clf = RandomizedSearchCV(svr_lin, para_grid)
    clf.fit(n_X.iloc[:-n_n].values,n_realized_vol.iloc[1:-(n_n-1)].values.reshape(-1,))
    
    n_predict_svr_lin = clf.predict(n_X.iloc[-n_n:])
    n_predict_svr_lin = pd.DataFrame(n_predict_svr_lin)
    n_predict_svr_lin.index = n_returns.iloc[-n_n:].index
    n_rmse_svr = np.sqrt(mse(n_realized_vol.iloc[-n_n:]/10000,n_predict_svr_lin/10000))
    errors = ((n_realized_vol['std'].iloc[-n_n:].reset_index()['std']) -(n_predict_svr_lin[0].reset_index()[0]))
    errors_n = ((errors - errors.mean())/errors.std())
    eks = kstest(list(errors_n),'norm')
    enormal = False
    if eks.pvalue >= .05:
        enormal=True
    ma = moving_average_inverse(n_X.iloc[-6:-1][1],n_predict_svr_lin[0].iloc[-1],1)/10000
    n_realized_vol.index = n_returns.iloc[5:].index
    ub =  n_predict_svr_lin[0].iloc[-1] + (errors.mean() + errors.std()*ci)
    lb = n_predict_svr_lin[0].iloc[-1] + (errors.mean() -errors.std()*1)
    upper = moving_average_inverse(n_X.iloc[-6:-1][1],ub,1)/10000
    lower = moving_average_inverse(n_X.iloc[-6:-1][1],lb,1)/10000
    if upper > lower:
        results_dict['Upper_Bound'].append(upper)
        results_dict['Lower_Bound'].append(lower)
    else:
        results_dict['Upper_Bound'].append(lower)
        results_dict['Lower_Bound'].append(upper)

    results_dict['symbol'].append(symbol)
    results_dict['Model'].append('SVR_Linear_Garch')
    results_dict['Date'].append(no_phenomena['date'].iloc[-1])
    results_dict['dayofweek'].append(i)
    results_dict['RMSE'].append(round(n_rmse_svr,6))
    results_dict['Last_Prediction'].append(n_predict_svr_lin[0].iloc[-1]/10000)
    results_dict['Inv_Ma'].append(ma)
    results_dict['Errors_ND'].append(enormal)
    results_dict['Last_Error'].append(errors.iloc[-1]/10000)
    results_dict['Error_Mean'].append(errors.mean()/10000)
    results_dict['Error_Std'].append(errors.std()/10000)


##    plt.figure(figsize=(10, 6))
##    plt.plot(n_realized_vol / 10000, label='Realized Volatility No Phenomena')
##    plt.plot(n_predict_svr_lin / 10000, label='Standard Deviation Prediction-SVR-GARCH No Phenomena')
##    plt.title(symbol+' '+str(dayofweek)+'_'+str(ci)+'stds No Phenomena Standard Deviation Prediction with SVR-GARCH (Linear)', fontsize=12)
##    plt.legend()
##    plt.savefig('Data/Garch_Results/Images/'+symbol+'_'+str(dayofweek)+'_'+str(ci)+'stds_No_Phenomena_Standard_Deviation_SVR-GARCH.png')
##    plt.close()
##    df= pd.DataFrame.from_dict(results_dict)
##    df.to_csv('Data/Garch_Results/'+symbol+'_'+str(dayofweek)+'_'+str(ci)+'stds_Intraday_Standard_Deviation_Garch_Results.csv',header=True)

def individual_garch_intraday_mean(symbol, dayofweek,xgb_pred,ci):
    results_dict = {'symbol':[],'Model':[],'Date':[],'dayofweek':[],'RMSE':[],'Last_Prediction':[],'Inv_Ma':[],'Errors_ND':[],'Last_Error':[],'Error_Mean':[],'Error_Std':[],'Upper_Bound':[],'Lower_Bound':[]}
    ## Data WITHOUT Holidays,Fed Dates, Earnings Dates
    no_phenomena = date_removal_intraday(symbol,dayofweek).reset_index().copy()
    no_phenomena['means'].loc[-1] = xgb_pred
    n_returns = no_phenomena['means']*100000
    n_realized_vol = n_returns.rolling(6).mean()
    n_n = int(round(len(no_phenomena['means'])*.3,0))
    n_split_date = n_returns.iloc[-n_n:].index
    ## SVR-Garch Linear No Phenomena
    n_realized_vol = pd.DataFrame(n_realized_vol)
    n_realized_vol.reset_index(drop=True, inplace=True)
    n_returns_svm = n_returns
    n_returns_svm = n_returns_svm.reset_index()
    
    del n_returns_svm['index']
    n_X = pd.concat([n_realized_vol, n_returns_svm], axis=1, ignore_index=True)
    n_X = n_X[5:].copy()
    n_X = n_X.reset_index()
    n_X.drop('index', axis=1, inplace=True)
    n_realized_vol = n_realized_vol.dropna().reset_index()
    n_realized_vol.drop('index', axis=1, inplace=True)
    svr_lin = SVR(kernel='linear')
    para_grid = {'gamma': sp_rand(),'C': sp_rand(),'epsilon': sp_rand()} 
    clf = RandomizedSearchCV(svr_lin, para_grid)
    clf.fit(n_X.iloc[:-n_n].values,n_realized_vol.iloc[1:-(n_n-1)].values.reshape(-1,))
    
    n_predict_svr_lin = clf.predict(n_X.iloc[-n_n:])
    n_predict_svr_lin = pd.DataFrame(n_predict_svr_lin)
    n_predict_svr_lin.index = n_returns.iloc[-n_n:].index
    n_rmse_svr = np.sqrt(mse(n_realized_vol.iloc[-n_n:]/100000,n_predict_svr_lin/100000))
    errors = ((n_realized_vol['means'].iloc[-n_n:].reset_index()['means']) -(n_predict_svr_lin[0].reset_index()[0]))
    errors_n = ((errors - errors.mean())/errors.std())
    eks = kstest(list(errors_n),'norm')
    enormal = False
    if eks.pvalue >= .05:
        enormal=True
    ma = moving_average_inverse(n_X.iloc[-6:-1][1],n_predict_svr_lin[0].iloc[-1],1)/100000
    n_realized_vol.index = n_returns.iloc[5:].index
    ub =  n_predict_svr_lin[0].iloc[-1] + (errors.mean() + errors.std()*ci)
    lb = n_predict_svr_lin[0].iloc[-1] + (errors.mean() -errors.std()*ci)
    upper = moving_average_inverse(n_X.iloc[-6:-1][1],ub,1)/100000
    lower = moving_average_inverse(n_X.iloc[-6:-1][1],lb,1)/100000
    if upper > lower:
        results_dict['Upper_Bound'].append(upper)
        results_dict['Lower_Bound'].append(lower)
    else:
        results_dict['Upper_Bound'].append(lower)
        results_dict['Lower_Bound'].append(upper)
    
    results_dict['symbol'].append(symbol)
    results_dict['Model'].append('SVR_Linear_Garch')
    results_dict['Date'].append(no_phenomena['date'].iloc[-1])
    results_dict['dayofweek'].append(i)
    results_dict['RMSE'].append(round(n_rmse_svr,6))
    results_dict['Last_Prediction'].append(n_predict_svr_lin[0].iloc[-1]/100000)
    results_dict['Inv_Ma'].append(ma)
    results_dict['Errors_ND'].append(enormal)
    results_dict['Last_Error'].append(errors.iloc[-1]/100000)
    results_dict['Error_Mean'].append(errors.mean()/100000)
    results_dict['Error_Std'].append(errors.std()/100000)

##    plt.figure(figsize=(10, 6))
##    plt.plot(n_realized_vol / 100000, label='Real Means No Phenomena')
##    plt.plot(n_predict_svr_lin / 100000, label='Mean Change Prediction-SVR-GARCH No Phenomena')
##    plt.title(symbol+' '+str(dayofweek)+'_'+str(ci)+'stds No Phenomena Mean Change Prediction with SVR-GARCH (Linear)', fontsize=12)
##    plt.legend()
##    plt.savefig('Data/Garch_Results/Images/'+symbol+'_'+str(dayofweek)+'_'+str(ci)+'stds_No_Phenomena_Mean_SVR-GARCH.png')
##    plt.close()
##    df= pd.DataFrame.from_dict(results_dict)
##    df.to_csv('Data/Garch_Results/'+symbol+'_'+str(dayofweek)+'_'+str(ci)+'stds_Intraday_Mean_Garch_Results.csv',header=True)


def individual_garch_close_over_close(symbol, dayofweek,xgb_pred,ci):
    results_dict = {'symbol':[],'Model':[],'Date':[],'dayofweek':[],'RMSE':[],'Last_Prediction':[],'Inv_Ma':[],'Errors_ND':[],'Last_Error':[],'Error_Mean':[],'Error_Std':[],'Upper_Bound':[],'Lower_Bound':[]}
    ## Data WITHOUT Holidays,Fed Dates, Earnings Dates
    no_phenomena = date_removal_close_over_close(symbol,dayofweek).reset_index().copy()
    no_phenomena['cumsum'].loc[-1] = xgb_pred
    n_returns = no_phenomena['cumsum']*1000
    n_realized_vol = n_returns.rolling(6).mean()
    n_n = int(round(len(no_phenomena['cumsum'])*.3,0))
    n_split_date = n_returns.iloc[-n_n:].index
    ## SVR-Garch Linear No Phenomena
    n_realized_vol = pd.DataFrame(n_realized_vol)
    n_realized_vol.reset_index(drop=True, inplace=True)
    n_returns_svm = n_returns
    n_returns_svm = n_returns_svm.reset_index()
    
    del n_returns_svm['index']
    n_X = pd.concat([n_realized_vol, n_returns_svm], axis=1, ignore_index=True)
    n_X = n_X[5:].copy()
    n_X = n_X.reset_index()
    n_X.drop('index', axis=1, inplace=True)
    n_realized_vol = n_realized_vol.dropna().reset_index()
    n_realized_vol.drop('index', axis=1, inplace=True)
    svr_lin = SVR(kernel='linear')
    para_grid = {'gamma': sp_rand(),'C': sp_rand(),'epsilon': sp_rand()} 
    clf = RandomizedSearchCV(svr_lin, para_grid)
    clf.fit(n_X.iloc[:-n_n].values,n_realized_vol.iloc[1:-(n_n-1)].values.reshape(-1,))
    
    n_predict_svr_lin = clf.predict(n_X.iloc[-n_n:])
    n_predict_svr_lin = pd.DataFrame(n_predict_svr_lin)
    n_predict_svr_lin.index = n_returns.iloc[-n_n:].index
    n_rmse_svr = np.sqrt(mse(n_realized_vol.iloc[-n_n:],n_predict_svr_lin))
    errors = ((n_realized_vol['cumsum'].iloc[-n_n:].reset_index()['cumsum']) -(n_predict_svr_lin[0].reset_index()[0]))
    errors_n = ((errors - errors.mean())/errors.std())
    eks = kstest(list(errors_n),'norm')
    enormal = False
    if eks.pvalue >= .05:
        enormal=True

    ma = moving_average_inverse(n_X.iloc[-6:-1][1],n_predict_svr_lin[0].iloc[-1],1)/1000
    ub =  n_predict_svr_lin[0].iloc[-1] + (errors.mean() + errors.std()*ci)
    lb = n_predict_svr_lin[0].iloc[-1] + (errors.mean() - errors.std()*ci)
    upper = moving_average_inverse(n_X.iloc[-6:-1][1],ub,1)/1000
    lower = moving_average_inverse(n_X.iloc[-6:-1][1],lb,1)/1000
    if upper > lower:
        results_dict['Upper_Bound'].append(upper)
        results_dict['Lower_Bound'].append(lower)
    else:
        results_dict['Upper_Bound'].append(lower)
        results_dict['Lower_Bound'].append(upper)
    results_dict['symbol'].append(symbol)
    results_dict['Model'].append('SVR_Linear_Garch')
    results_dict['Date'].append(no_phenomena['timestamp'].iloc[-1])
    results_dict['dayofweek'].append(dayofweek)
    results_dict['RMSE'].append(round(n_rmse_svr,6))
    results_dict['Last_Prediction'].append(n_predict_svr_lin[0].iloc[-1]/1000)
    results_dict['Inv_Ma'].append(ma)
    results_dict['Errors_ND'].append(enormal)
    results_dict['Last_Error'].append(errors.iloc[-1]/1000)
    results_dict['Error_Mean'].append(errors.mean()/1000)
    results_dict['Error_Std'].append(errors.std()/1000)

    n_realized_vol.index = n_returns.iloc[5:].index

##    plt.figure(figsize=(10, 6))
##    plt.plot(n_realized_vol / 1000, label='Real Close over Close change No Phenomena')
##    plt.plot(n_predict_svr_lin / 1000, label='Close over close Prediction-SVR-GARCH No Phenomena')
##    plt.title(symbol+' '+str(dayofweek)+'_'+str(ci)+'stds No Phenomena Close_over_Close Prediction with SVR-GARCH (Linear)', fontsize=12)
##    plt.legend()
##    plt.savefig('Data/Garch_Results/Images/'+symbol+'_'+str(dayofweek)+'_'+str(ci)+'stds_No_Phenomena_Close_over_Close_SVR-GARCH.png')
##    plt.close()
##
##    df= pd.DataFrame.from_dict(results_dict)
##    df.to_csv('Data/Garch_Results/'+symbol+'_'+str(dayofweek)+'_'+str(ci)+'stds_Close_over_Close_Garch_Results.csv',header=True)

def close_over_close_xgb_svr_predictions(symbol,dayofweek,ci):
    no_phenomena =  date_removal_close_over_close(symbol,dayofweek)
    n_returns = list(no_phenomena['cumsum'])
    values = n_returns[:-1]
    train = series_to_timesequence(values, n_in=1)
    # transform the time series data 
    trainX, trainy = train[:, :-1], train[:, -1]

    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)

    row = values[-1:]

    yhat = model.predict(np.asarray([row]))

    individual_garch_close_over_close(symbol,dayofweek, yhat[0],ci)
    
def open_xgb_srv_predictions(symbol,dayofweek,ci):
    no_phenomena =  date_removal_open(symbol,dayofweek)
    n_returns = list(no_phenomena['change'])
    values = n_returns[:-1]
    train = series_to_timesequence(values, n_in=1)
    # transform the time series
    trainX, trainy = train[:, :-1], train[:, -1]

    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    
    row = values[-1:]

    yhat = model.predict(np.asarray([row]))

    individual_garch_intraday_open(symbol,dayofweek, yhat[0],ci)

def mean_xgb_srv_predictions(symbol,dayofweek,ci):
    no_phenomena =  date_removal_intraday(symbol,dayofweek)
    n_returns = list(no_phenomena['means'])
    values = n_returns[:-1]
    train = series_to_timesequence(values, n_in=1)
    # transform the time series
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)

    row = values[-1:]

    yhat = model.predict(np.asarray([row]))
    individual_garch_intraday_mean(symbol,dayofweek, yhat[0],ci)

def std_xgb_srv_predictions(symbol,dayofweek,ci):
    no_phenomena =  date_removal_intraday(symbol,dayofweek)
    n_returns = list(no_phenomena['std'])
    values = n_returns[:-1]
    train = series_to_timesequence(values, n_in=1)
    # transform the time series 
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)

    row = values[-1:]
 
    yhat = model.predict(np.asarray([row]))
    individual_garch_intraday_std(symbol,dayofweek, yhat[0],ci)

##for symbol in ["AMD","GOOG"]:
##    for i in range(0,5,1):
##        for j in range(1,3,1):
##            std_xgb_srv_predictions(symbol,i,j)
##            open_xgb_srv_predictions(symbol,i,j)
##            close_over_close_xgb_svr_predictions(symbol,i,j)
##            mean_xgb_srv_predictions(symbol,i,j)
