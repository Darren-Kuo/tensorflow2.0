import numpy as np
import pandas as pd


class Data_Processing(object):
    def Get_Data(self, date_from, date_to, period, mongo_client):
        minute = pd.DataFrame(list(mongo_client.find({'Time' : {'$gte' : date_from, '$lte':date_to}}, {'_id': False})))
        minute = minute.set_index('Time').sort_index()

        ohlcv_dict = {'Open' : 'first',
                      'High' : 'max',
                      'Low' : 'min',
                      'Close' : 'last',
                      'Volume' : 'sum'}

        new_data = minute.resample(period, closed='right', label='right').agg(ohlcv_dict).dropna(axis=0, how='any')

        return new_data
    
    def Normalization(self, df):
        _df = df.copy()
        df_norm = _df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

        return df_norm
    
    def Get_Label(self, df):
        _df = df.copy()
        _df['Change'] = _df.Close.shift(-3) - _df.Close
        _df['Label'] = _df.Change.apply(lambda x: 1 if x >= 20 else (2 if x <= -20 else 0))
        df_label = _df['Label']

        return df_label
    
    def Build_Train(self, df_x, df_y, pastDay = 20):
        list_x, list_y = [], []

#         df_y = pd.Series(df_y['Label'].values)

        for i in range(df_x.shape[0]-pastDay):
            list_x.append(np.array(df_x.iloc[i:i+pastDay]))
            list_y.append(np.array(df_y.iloc[i + pastDay - 1]))

        return np.array(list_x), np.array(list_y)
    
    
    def Shuffle_Data(self, X, Y):
        np.random.seed(10)
        randomList = np.arange(X.shape[0])
        np.random.shuffle(randomList)
        
        return X[randomList], Y[randomList]


    def Split_Data(self, X, Y, rate):
        X_train = X[int(X.shape[0]*rate):]
        Y_train = Y[int(Y.shape[0]*rate):]
        X_val = X[:int(X.shape[0]*rate)]
        Y_val = Y[:int(Y.shape[0]*rate)]

        return X_train, Y_train, X_val, Y_val