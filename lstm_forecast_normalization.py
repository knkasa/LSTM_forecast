# LSTM paper 
# https://marketpsych-website.s3.amazonaws.com/web3/files/papers/Forecasting%20the%20USD-JPY%20Rate%20with%20Sentiment.pdf

import sys
import datetime as dt
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import figure
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import yfinance as yf
import sqlalchemy as sa
import shap
import pandas_ta as ta
from sklearn.model_selection import KFold

from dataclasses import dataclass
from marketpsych import sftp

os.chdir('C:/my_working_env/general_folder/Refinitiv/refinitiv_lstm_test')
rms_freq = "W-SUN"      # News data time step.  W-SUN=every_Sunday  B=business_day, D=daily
price_freq = "1wk"  # "1wk"   # Yahoo finance price data time step.   1wk=every_week  1d=daily
activation_func =   [ 'relu', ]    #  ['relu', 'tanh']   
train_verbose = False
use_best_model = True
my_batch_size = None  # use None=len(df) or integer.  
num_folds = 5  # number of train data to split during k-cross validation.  
rma_sign =  -1.0   #  Sign of news data for plotting.    

normalization = "maxmin"   # Choose log, maxmin, ...
n_steps =  5   # Lookup time for next predictions.  
use_macd = "sentiment"    # Apply macd to rma data.  Choose which column to apply macd.  
reduceLR = True   # Option to change learning rate during training.

num_units = [ 100 ]   # [ 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140   ]  
num_layers =    [   2,  ]  

cur =   "USDJPY"    #"GBPUSD"  #"USDJPY"
rma_file = "rma_colab_USD.csv"   # "rma_colab_GBP.csv"     # rma_colab_USD.csv  rma_colab_JPY.csv  rma_colab_GBP.csv    
rma_file2 = ""   #"rma_colab_USD.csv" 

# Columns for the model.    
# sentiment, sentiment_x, sentiment_y, rma1_macd, rma1_diff, rma1_signal, rma2_macd, rma2_diff, rma2_signal
features =   [    'Vol',   "sentiment",  "rma1_diff",   ]      # Add _x _y if second rma file is enabled.   "rma2_macd",  "rma2_signal", "rma2_diff", 
rename_features = [   'Vol',  "sentiment_US",    "macd_diff_US",        ]   # "sentiment_GB",  "sentiment_JP",    "macd_diff_GB",   "macd_diff_JP",    

#-----------------------------------------------------------------------------

# Retrieves weekly price data from yfinance
price_df = yf.download( cur + "=X", interval=price_freq, start="1997-12-29", end="2022-02-27", progress=False)
# Replaces an incorrect price data point with a rough estimate
if cur=="USDJPY":  price_df.loc['2007-11-19', 'High'] = 111  # Fix the bug, the price was greater than 600.
price_df.to_csv('./price.csv')

# Get prepared news data from google colab. 1998-Jan-1~2022-Feb-28 daily.  
rma = pd.read_csv(rma_file)
rma.rename(columns = {'windowTimestamp':'utc_datetime'}, inplace=True)
rma['utc_datetime'] = pd.to_datetime( rma.utc_datetime, utc=False ) 
rma.fillna(0, inplace=True)
if use_macd:
    if normalization=="log":
    #if n_steps == 1:
        v=12*1
    elif normalization=="maxmin":
    #else:
        v=12*4
    rma[['rma1_macd','rma1_diff','rma1_signal']] = ta.macd( rma[use_macd], fast=v, slow=int(v*26/12), signal=int(v*9/12) ).fillna(0)*20.0
    
if rma_file2:
    rma2 = pd.read_csv(rma_file)   
    rma2.rename(columns = {'windowTimestamp':'utc_datetime'}, inplace=True)
    rma2['utc_datetime'] = pd.to_datetime( rma.utc_datetime ) 
    rma2.fillna(0, inplace=True)
    if use_macd:
        rma2[['rma2_macd','rma2_diff','rma2_signal']] = ta.macd( rma2[use_macd], fast=v, slow=int(v*26/12), signal=int(v*9/12) ).fillna(0)*20.0

#---------- plot MACD -----------------------------------------------
'''
test_df = rma[['utc_datetime','rma1_macd','rma1_signal']].copy() 
test_df['date'] = test_df['utc_datetime'].dt.date
price_df.reset_index( drop=False, inplace=True )
price_df['date'] = price_df.Date.dt.date
test_df = pd.merge(  price_df  , test_df, on='date', how='left')      #join.  left=left_join, inner=inner_join

test_df = test_df.loc[ 50:250 ] 
fig, ax = plt.subplots(figsize=(6, 5))
test_df.plot( x='date', y='Open', ax=ax, label="daily open")
ax.legend(loc='upper left')
ax.set_ylabel( 'price', )

ax = ax.twinx()
test_df.plot( x='date', y='rma1_macd', ax=ax, color='green', alpha=0.5, label= 'macd' )
test_df.plot( x='date', y='rma1_signal', ax=ax, color='red', alpha=0.5, label= 'signal' )
ax.set_ylabel( 'macd', rotation=270, labelpad=5)
ax.legend(loc='upper right')
fig.autofmt_xdate(rotation=20)
plt.show()
plt.close()
#exit()  
'''
#---------------------------------------------------------------------

# Calculates volatility based on high/low prices
def calc_vol_hl(prices_df):
    return  np.sqrt((np.log(prices_df.High/prices_df.Low))**2)/(2.0*np.log(4))  # Note this is same as  np.log(prices_df.High/prices_df.Low)/(2.0*np.log(4))

def buzz_weight_rmas( rma_df: pd.DataFrame ) -> pd.DataFrame:
    '''
    Re-aggregates RMAs within a given time frequency. It also flattens the 
    dataframes so the assets become different columns (if more than one currency is in the data).
    
    freq: frequency to combine by 
        e.g. '3H' for 3 hours, 'W-FRI' for week ending on friday, 'M' for calendar month
        see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases    
    '''
    
    buzz_col="buzz"
    asset_col="assetCode"
    date_col="utc_datetime"   #"windowTimestamp"

    if date_col in rma_df.columns:
        rma_df[date_col] = pd.to_datetime(pd.to_datetime(rma_df.utc_datetime).dt.strftime('%Y-%m-%d'))
        rma_df = rma_df.set_index(date_col)

    cols_to_ignore = [buzz_col, "mentions", "utc_datetime",   #"windowTimestamp", 
                      "dataType", "systemVersion", "id", "Date"]
    
    if asset_col in rma_df.columns:
        assets = rma_df[asset_col].unique()  # Find list of currency names.  
        
        final_df = None
        for asset in assets:  # loop through different currencies.  
            asset_df = rma_df.query(f"{asset_col} == '{asset}'").drop(columns=[asset_col]).copy()
            
            # Aggregates buzz on the chosen time period
            temp_df = asset_df[[buzz_col]].groupby(pd.Grouper(freq=rms_freq)).sum().copy()

            for rma in asset_df.columns:
                if rma not in cols_to_ignore:
                    # Computes the raw counting of the rma.  
                    temp_df[rma] = (asset_df[rma]*asset_df[buzz_col]).groupby(pd.Grouper(freq=rms_freq)).sum()   
                    temp_df[rma] = temp_df[rma]/temp_df[buzz_col]    # Renormalizes by the total buzz
                        
            if final_df is not None:
                final_df = pd.concat([final_df, temp_df], axis=1)
            else:
                final_df = temp_df.copy()
        
        return final_df

# Computes volatility and price changes (log)
price_df['Vol'] = calc_vol_hl(price_df).shift()

# Normalize price.
if normalization == "log":
    rate_col_name = "rate_log"
    price_df['rate_log'] = np.log(price_df.Open/price_df.Open.shift())*10.0    # log(price) - log(price_day_before)
    price_df['Fw_rate_log'] = price_df.rate_log.shift(-1) 
elif normalization == "maxmin":
    if cur == "USDJPY":
        max0 = 150; min0 = 100
    elif cur == "GBPJPY":
        max0 = 150; min0 = 100
    elif cur == "GBPUSD":
        max0 = 1.4; min0 = 1.1
    elif cur == "EURUSD:
        max0 = 1.2; min0 = 1.0
    rate_col_name = "rate_maxmin"
    price_df[rate_col_name] = (price_df.Open - 100)/(150-100)   # max-min normzalization (price-min)/(max-min)
    price_df['Fw_' + rate_col_name] = price_df.rate_maxmin.shift(-1)

price_df.to_csv('./price_df.csv')

# Re-aggregates the RMAs into weekly values from Sunday to Sunday
rma_df = rma.copy()
rma_df = buzz_weight_rmas(rma_df)
rma_df.to_csv('C:/my_working_env/general_folder/Refinitiv/rma_df.csv')
#print( rma_df ) 
#print( price_df ) 

# Make sure it merged correctly.
combined_df = pd.merge_asof(price_df, rma_df, left_index=True, right_index=True)

if rma_file2:
    rma_df2 = rma2.copy()
    rma_df2 = buzz_weight_rmas(rma_df2)
    combined_df = pd.merge_asof(combined_df, rma_df2, left_index=True, right_index=True)

combined_df.to_csv('./combined_df.csv')

#--------------- Plot --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
combined_df["Open"].plot(ax=ax, label="Weekly open")
ax.legend(loc='upper left')
ax.set_ylabel(cur)
ax.grid(False)
#ax.set_title(f"USDJPY vs the inverse of the JPY sentiment, corr: \
#   {combined_df[['LogRet','sentiment_JPY']].corr()['LogRet'].iloc[1]:.1%}")
ax.set_title( rma_file[-7:-4] + "  " + features[1] )

rma_feature = features[1]  # plot the last column.
ax2 = ax.twinx()
(rma_sign*combined_df[rma_feature]).plot(ax=ax2, color='red', alpha=0.5, label= str(rma_sign)+"*rma" )
ax2.grid(False)
ax2.set_ylabel( str(rma_sign) + "*rma", rotation=270, labelpad=5)
ax2.legend(loc='upper right')
#plt.show(); exit()
plt.close()

#------------ prepare train, test data ---------------------------------

feature_list = [rate_col_name] + features
combined_df = combined_df[ feature_list ]
if "rma1_diff" in combined_df.columns:
    combined_df.drop(  combined_df[ (combined_df.rma1_diff==0) | (combined_df.rma1_diff.isnull())].index , inplace=True )

combined_train = combined_df['1998':'2018'].fillna(0).copy()
combined_test = combined_df['2019':].fillna(0).copy()
#combined_is = combined_df[:int(len(combined_df)*0.8)].fillna(0).copy()
#combined_os = combined_df[int(len(combined_df)*0.8)+1:].fillna(0).copy()

def split_sequences(sequences, n_steps):
    
    # From https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
    X, y = list(), list()
    for i in range(len(sequences)):
        
        # find the end of this pattern
        end_ix = i + n_steps
        
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
         
        # Get input and target data. Target is 1 day ahead of input data.
        seq_x = sequences[i:end_ix, :]
        seq_y = sequences[end_ix, :]   
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Reshape data into NxTxD  T=lookup_time,  D=num_columns
# y_train is the target data (one time step ahead of X_train)
X_train, y_train = split_sequences( combined_train.fillna(0).to_numpy(), n_steps)
X_test, y_test = split_sequences( combined_test.fillna(0).to_numpy(), n_steps)

# Selects LogRet column as target data.  Dont need other columns as target.  
label_index = 0    # 0 = LogRet column
y_train = y_train[:, label_index]
y_test  = y_test[:, label_index]

# Creates an ouptut directory for holding results and temp models.
ROOT_DIR = "."
folder_name = 'output'
if not os.path.exists(folder_name):
    os.makedirs(os.path.join(ROOT_DIR, folder_name))
OUT_DIR = os.path.join(ROOT_DIR, folder_name)

#---------- define neural network ---------------------------------------------------
@dataclass # <-- this is same as def __init__(self):  https://realpython.com/python-data-classes/
class MarketPsychLSTM(object):

    # Default Network settings
    dropout     = 0.2
    loss        = 'mse'
    optimizer   = 'adam'

    # Early stop settings
    es_monitor  = 'val_loss'
    es_mode     = 'min'
    es_patience = 50
    es_verbose  = 0
    mc_filename = 'best_model.h5'
    lr_patience = int(es_patience/2)

    # Fit settings
    max_epoch      = 1000   
    batch_size  = 64
    verbose     = 0
    shuffle     = False
    
    def _build_network(self, X):

        model = tf.keras.models.Sequential()
        if self.nlayers == 0:
            raise ValueError('Number of layers must be at least 1')
        elif self.nlayers == 1:
            model.add(tf.keras.layers.LSTM(self.nunits))
        else:
            for layer in range(self.nlayers-1):
                # Default activation is tanh
                model.add(tf.keras.layers.LSTM(self.nunits, activation=self.activation, 
                              return_sequences=True, 
                              input_shape=(X.shape[1], X.shape[2])))
                model.add(tf.keras.layers.Dropout(self.dropout))
            # Add last hidden layer
            model.add(tf.keras.layers.LSTM(self.nunits))
            
        # Output layer
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        # Creates object with model
        self.model = model

    # Early stopping configuration
    def _early_stopping(self):
        es = tf.keras.callbacks.EarlyStopping(monitor=self.es_monitor, mode=self.es_mode, 
                              verbose=self.es_verbose, patience=self.es_patience)
        mc = tf.keras.callbacks.ModelCheckpoint(os.path.join(OUT_DIR, self.mc_filename), 
                            monitor=self.es_monitor, mode=self.es_mode, 
                            verbose=0, save_best_only=True)
        rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", 
                                                    patience=self.lr_patience, verbose=0)
        return es, mc, rlr
    
    def fit_once(self, X, y, val=None):  # used for SHAP
        self._build_network(X)
        if val is not None:
            es, mc = self._early_stopping()
            if reduceLR:
                callbacks = [es, mc, rlr]
            else:
                callbacks = [es, mc]
        else:
            callbacks = None
        
        history = self.model.fit(X, y, epochs=self.max_epoch, 
                                batch_size=self.batch_size, 
                                use_multiprocessing=True,
                                validation_data=val,
                                verbose=self.verbose,
                                shuffle=self.shuffle,
                                callbacks=callbacks)
        self.history = history
        
        if val is not None:
            self.model = tf.keras.models.load_model(os.path.join(OUT_DIR, self.mc_filename))

    def predict(self, X):
        return self.model.predict(X)
    
    def fit_kfold(self, X, y, num_folds, kfold_verbose=1):
    
        # Builds network
        self._build_network(X)

        # Series with predictions in the validation set
        valid_series  = pd.Series([], dtype='float64')
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=num_folds, shuffle=False)

        es, mc, rlr = self._early_stopping()
        fold_no = 1
        mse_per_fold = []
        opt_n_epochs = []
        for train, valid in kfold.split(X, y):
            if kfold_verbose:
                print('----------------------------')
                print(f'Training fold {fold_no} ...')

            if reduceLR:
                callback_opt = [es, mc, rlr]
            else:
                callback_opt = [es, mc]

            history = self.model.fit(X[train], y[train],
                                    epochs=self.max_epoch, 
                                    batch_size=self.batch_size, 
                                    use_multiprocessing=True,
                                    validation_data=(X[valid], y[valid]),
                                    verbose=self.verbose,
                                    shuffle=False, callbacks=callback_opt)
        
            # Load the best saved model.
            saved_model = tf.keras.models.load_model(os.path.join(OUT_DIR, self.mc_filename))
            self.model = saved_model
            opt_epochs = len(history.history['val_loss'])
            opt_n_epochs.append(opt_epochs)
            if kfold_verbose:
                print(f'Epochs used: {opt_epochs}')

            scores = saved_model.evaluate(X[valid], y[valid], verbose=0)
            
            if kfold_verbose:
                print('RMSE for fold {:d}: {:>7.4}'.format(fold_no, np.sqrt(scores)))
            mse_per_fold.append(scores)
            
            temp_sr = pd.Series(np.squeeze(saved_model.predict(X[valid], verbose=0)), index=valid)
            valid_series = pd.concat([valid_series, temp_sr])

            # Increase fold number
            fold_no = fold_no + 1

        mse_per_fold = np.array(mse_per_fold)
        if kfold_verbose:
          print('Avg. RMSE: {:>7.4f}'.format(np.sqrt(mse_per_fold).mean()))
        return mse_per_fold, opt_n_epochs, valid_series

def plot_lstm_history(history, init_step=0):
    # Plot history
    for key in history.history.keys():
        plt.plot(history.history[key][init_step:], label=key)
    plt.xlabel('# epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

#--------------- start training ------------------------------------------
tf.random.set_seed(52)
#tf.keras.utils.set_random_seed(52)

model = MarketPsychLSTM()
if my_batch_size == None:
    model.batch_size = len(X_train)
elif isinstance( my_batch_size, int ):
    model.batch_size = my_batch_size
else:
    raise exception("batch needs to be an integer")

rmse_list = [ 1.0e10 ]
for nunits in num_units:
    for nlayers in num_layers:
        for activation in activation_func:
            model.nunits = nunits
            model.nlayers =  nlayers
            model.activation = activation

            # Runs a few times due to the stochastic character of the model
            mse_folds, _, _ = model.fit_kfold(X_train, y_train, num_folds, kfold_verbose=train_verbose)
            
            mean_rmse = np.mean(np.sqrt(np.array(mse_folds)))
            print(f" Num_units = {str(nunits)},  Num_layers = {str(nlayers)}, Activation = {activation}, AVG_RMSE = {mean_rmse}")
            if mean_rmse < min(rmse_list):
                best_model = { 'Num_units':nunits, 'Num_layers':nlayers, 'Activation':activation }
            
            rmse_list.append(mean_rmse)
            
print();  print(" Best model is: ", best_model )

#------------- Backtest result ------------------------------------------------------

def get_pips_result( df, pred_series ):

    def get_pips(x):
        if x.pred_position != x.pred_position_prev:
            if x.pred_position==-1.0:
                return x.Open - x.order_prev
            elif x.pred_position==1.0:
                return x.order_prev - x.Open
        else:
            return 0.0  

    def paper_pips_fun(x):
        if x.paper_position == 1:
            return x.Open - x.open_prev
        elif x.paper_position == -1:
            return x.open_prev - x.Open
        else:
            return 0.0    
    
    if normalization == "log":
        df['prediction'] = np.pad( pred_series, (n_steps-1, 1), constant_values=0)    
        df['pred_position'] = np.sign( df['prediction'] )
    elif normalization == "maxmin":
        df['prediction'] = np.pad( pred_series, (n_steps-1, 1), constant_values=0)
        df['pred_diff'] = df[['rate_maxmin','prediction']].apply( lambda x: x[1]-x[0], axis=1 )
        df['pred_position'] = np.sign( df['pred_diff'] ) 
        
    # keep position until sar_buy/sell
    df['pred_position_prev'] = df.pred_position.shift(periods=1)
    df['order_price'] = df[[ 'pred_position', 'pred_position_prev', 'Open' ]].apply(lambda x: x[2] if x[0]!=x[1] else  np.nan , axis=1)
    df.fillna( method='ffill', inplace=True )
    df['order_prev'] = df.order_price.shift(periods=1)
    df['pips'] = df[['pred_position', 'pred_position_prev', 'Open', 'order_prev']].apply(lambda x: get_pips(x), axis=1)
    df['cum_pips'] = df.pips.cumsum()

    # paper result.  Close position everyday.  
    if normalization == "log":
        df['paper_pred_shifted'] = np.pad( pred_series, (n_steps, 0), constant_values=0 )   # shift predicted log return to next day.
        df['paper_position'] = np.sign( df['paper_pred_shifted']  )   # position is the prediction from previous day.  
        df["logpips"] = df["paper_position"] * df["rate_log"]
        df["paper_logReturn"] = np.exp(df.logpips.cumsum())
    elif normalization == "maxmin":
        df['paper_position'] = df.pred_position.shift(periods=1)
    
    df["open_prev"] = df.Open.shift(periods=1)
    df['paper_pips'] = df[['paper_position', 'Open', 'open_prev']].apply(lambda x: paper_pips_fun(x), axis=1)   # close position everyday.
    df['cum_paper_pips'] = df.paper_pips.cumsum()
    
    # Insert win column.
    df['win'] = df[['paper_pips']].apply( lambda x: 1 if np.sign(x.paper_pips)>0 else 0, axis=1 )
        
    return df

# Based on the above result, the lowest rmse given was the following params.  
if use_best_model:
    subtitle = " Using best params"
    model.nunits = best_model['Num_units']
    model.nlayers =  best_model['Num_layers']   
    model.activation = best_model['Activation']
else:
    subtitle = "Using default params"
    model.nunits = 15  
    model.nlayers =  2   
    model.activation = "relu"

tf.random.set_seed(52)

# Get Open price in data.  
combined_train = pd.merge_asof( combined_train,  price_df['Open'], left_index=True, right_index=True )

# Reruns the model with the chosen configuration.
# yhat_train is the predicted LogReturn using validation data.
print();  print( subtitle + " to train the model again...")
_, _, pred_train = model.fit_kfold(X_train, y_train, num_folds, kfold_verbose=False)  
combibned_train = get_pips_result( combined_train, pred_train.values )

# plot cum_pips, paper_logReturn, cum_paper_pips.  paper_logReturn reproduces the paper.  
combined_train.to_csv('./output_train.csv')


#------------ Now for the test data. --------------------------------------
print("Working on the test data ...")
combined_test = pd.merge_asof( combined_test,  price_df['Open'], left_index=True, right_index=True )

pred_test = model.predict(X_test)

combibned_test = get_pips_result( combined_test, np.squeeze(pred_test) )
combined_test.to_csv('./output_test.csv')

#------------- Plot validation and forward result --------------------------

font_size = 14
win_pct_val = np.round( combined_train.win.sum()/(len(combined_train)-1) , 2 ) 
win_pct_forward = np.round( combined_test.win.sum()/(len(combined_test)-1) , 2 ) 
print(); print(f" win_pct(val) = {str(win_pct_val)},  win_pct(forward) = {str(win_pct_forward)}")

figure(num=None, figsize=(9, 5), dpi=80, facecolor='w', edgecolor='k')  
plt.plot( combined_train.index.values, combined_train.cum_paper_pips.values )
plt.title( cur + " (Validation)  " +  ' '.join(rename_features[1:]),  fontsize=font_size+2 )
plt.xticks( fontsize=font_size, rotation=20 );
plt.yticks( fontsize=font_size ) 
plt.ylabel( 'pips',  fontsize=font_size)
#plt.text( combined_train.index[10] , 5 , 'win_pct = ' + str(win_pct_val), fontsize=12 )
plt.grid()
plt.show()
plt.savefig('./validation.png' ,  dpi=200)
plt.close()

figure(num=None, figsize=(9, 5), dpi=80, facecolor='w', edgecolor='k') 
plt.plot( combined_test.index.values, combined_test.cum_paper_pips.values )
plt.title( cur + " (Forward)  " +  ' '.join(rename_features[1:]), fontsize=font_size+2 )
plt.xticks( fontsize=font_size, rotation=20 );
plt.yticks( fontsize=font_size ) 
plt.ylabel( 'pips', fontsize=font_size)
#plt.text( combined_test.index[10] , 5 , 'win_pct = ' + str(win_pct_forward), fontsize=12 )
plt.grid()
plt.show()
plt.savefig('./forward.png' ,  dpi=200)
plt.close()

#----------- Interpreting features --------------------------------------------
# Use of SHAP.  shap_value = It shows how much the feature has shifted predicted value away from the target value.  
# Blue means shifted in negative direction while Red means shifted in positive direction.
# https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137

# SHAP not currently working with TF >2.5.  Version needs to be adjusted.
tf.compat.v1.disable_v2_behavior()
tf.random.set_seed(52)

model.fit_once(X_train, y_train)

shap.initjs()
explainer = shap.DeepExplainer(model.model, X_train)
shap_values = explainer.shap_values(X_test)

features =  [rate_col_name] + rename_features

if n_steps >= 2:
    # https://stackoverflow.com/questions/60894506/shap-lstm-keras-tensorflow-valueerror-shape-mismatch-objects-cannot-be-broa
    shap.summary_plot( np.squeeze(shap_values)[:, 0, :], np.squeeze(X_test)[:, 0, :], features )
    shap.summary_plot( np.squeeze(shap_values)[:, 0, :], np.squeeze(X_test)[:, 0, :], features, plot_type='bar' )
else:
    shap.summary_plot( np.squeeze(shap_values), np.squeeze(X_test), features )
    shap.summary_plot( np.squeeze(shap_values), np.squeeze(X_test), features, plot_type='bar' )
#shap.summary_plot( np.squeeze(shap_values), np.squeeze(X_test), features, plot_type='bar' )

#explainer = shap.DeepExplainer(model.model, X_train)
#shap_values = explainer.shap_values(X_train)
#shap.summary_plot( shap_values[0].reshape(shap_values[0].shape[0], 3), X_train.reshape(X_train.shape[0], 3), features )
#shap.summary_plot( shap_values[0].reshape(shap_values[0].shape[0], 3), X_train.reshape(X_train.shape[0], 3), features,  plot_type='bar' )
#shap.force_plot(explainer.expected_value[0], shap_values[0][0], features)

import pdb; pdb.set_trace()



# TODO:
# test various input data with news(or, bloomberg, yahoo) using different rate (FX, stocks, crypto,)
# test without sentiment.
# test Tiago's suggested features.  
# increase lookup days. 
# test with daily data.  
# use different standardization (MACD, ...)  
# try N-beats, transformer,  ...  
# change column name (remove *JPY )
