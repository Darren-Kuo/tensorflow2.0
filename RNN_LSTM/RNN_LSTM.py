import tensorflow as tf
from tensorflow.keras import layers

from pymongo import MongoClient
from datetime import datetime as dt

from Backtesting import Backtesting
from Data_Processing import Data_Processing

DB_URL = ''
DB_NAME = ''
COL_NAME = ''

D_FROM = dt.strptime('2017-09-01 00:00:00', '%Y-%m-%d %H:%M:%S')
D_TO = dt.strptime('2019-06-01 00:00:00', '%Y-%m-%d %H:%M:%S')
LAST_PROFIT = 0


data_process = Data_Processing()
backtesting = Backtesting()

client = MongoClient(DB_URL)[DB_NAME][COL_NAME]
    
data = data_process.Get_Data(D_FROM, D_TO, '1H', client)
data['ma5'] = data.Close.rolling(5).mean()
data['ma10'] = data.Close.rolling(10).mean()
data['ma15'] = data.Close.rolling(15).mean()
data['ma20'] = data.Close.rolling(20).mean()
data['ma40'] = data.Close.rolling(40).mean()
data['ma60'] = data.Close.rolling(60).mean()

_data = data.tail(-60)

start = '2017-09-01'
end = '2018-12-31'
mask = (_data.index > start) & (_data.index <= end)
df_train = _data.loc[mask]

start = '2019-01-01'
end = '2019-07-01'
mask = (_data.index > start) & (_data.index <= end)
df_test = _data.loc[mask]


def df_to_dataset(df, shuffle=True, batch_size=32):
    labels = data_process.Get_Label(df)
    df_norm = data_process.Normalization(df)
    
    df_x, df_y = data_process.Build_Train(df_norm, labels)
    df_y = tf.keras.utils.to_categorical(df_y, 3)

    ds = tf.data.Dataset.from_tensor_slices((df_x, df_y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    
    return ds, df_x, df_y


batch_size = 200
train_ds, _, _ = df_to_dataset(df_train, batch_size=batch_size)
test_ds, test_x, test_y = df_to_dataset(df_test, shuffle=False, batch_size=batch_size)

train_ds = train_ds.repeat()


inputs = tf.keras.Input(shape=(20,11,), name='txf')
x = layers.LSTM(20, activation = 'tanh')(inputs)
x = layers.Dense(20, activation='relu')(x)
x = layers.Dense(10, activation='relu')(x)
outputs = layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


for i in range(10):
    # model.compile(optimizer=tf.keras.optimizers.Adam(0.01),loss='categorical_crossentropy')
    model.compile(optimizer='adam',loss='categorical_crossentropy')

    history = model.fit(train_ds, epochs = 10, steps_per_epoch = 10)

    pred = model.predict(test_ds)

    profit = backtesting.Backtest(df_test.tail(-20), tf.argmax(pred, axis = 1).numpy())

    if profit > LAST_PROFIT:
        model_name = './txf_1hr_lstm_%s.h5' % i
        detail_name = './detail_txf_1hr_lstm_%s.csv' % i

        model.save(model_name)
        backtesting.Save_Detail(detail_name)

        LAST_PROFIT = profit
    
    model.reset_states()