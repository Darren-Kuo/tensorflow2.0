import pandas as pd

AMOUNT = 1
POINT_PRICE = 200
FEES = 200

OpenTime = 'open_time'
OpenPrice = 'open_price'
CloseTime = 'close_time'
ClosePrice = 'close_price'
Volume = 'volume'
Side = 'side'
Profit = 'profit'


class Backtesting(object):
    def __init__(self):
        self.strike = {}
        self.detail = []
        self.last_signal = 'N'
        self.has_pos = False


    def Backtest(self, df, y_pred):
        _df_test = df.reset_index()

        _signal = pd.DataFrame(y_pred, columns = ['Signal']).shift(1)
        
        _df_test = pd.concat([_df_test, _signal], axis = 1).tail(-1)
        
        size = len(_df_test)
        
        for i in range(size):
            self.Trading(_df_test.iloc[i])            
            
        profit_list = [d[Profit] for d in self.detail]
        profit = sum(profit_list)
            
        return profit
            

    def Trading(self, data):
        _sig = data.Signal
        if not self.has_pos:
            if _sig == 1:
                self.Record_Pos(data, 'buy')
            elif _sig == 2:
                self.Record_Pos(data, 'sell')
        else:
            if _sig == 1 and self.strike[Side] == 'sell':
                self.Record_Detail(data)
                self.Record_Pos(data, 'buy')
            elif _sig == 2 and self.strike[Side] == 'buy':
                self.Record_Detail(data)
                self.Record_Pos(data, 'sell')


    def Record_Pos(self, data, side):       
        self.strike[OpenTime] = data.Time
        self.strike[OpenPrice] = data.Open
        self.strike[Volume] = AMOUNT
        self.strike[Side] = side
        self.has_pos = True
        
        
    def Record_Detail(self, data):
        self.strike[CloseTime] = data.Time
        self.strike[ClosePrice] = data.Open
        
        if self.strike[Side] == 'buy':
            self.strike[Profit] = (self.strike[ClosePrice] - self.strike[OpenPrice]) * POINT_PRICE - FEES
        elif self.strike[Side] == 'sell':
            self.strike[Profit] = (self.strike[OpenPrice] - self.strike[ClosePrice]) * POINT_PRICE - FEES
            
        self.detail.append(self.strike)
        self.strike = {}
        self.has_pos = False
    

    def Save_Detail(self, file_name):
        if len(self.detail) > 0:
            df_detail = pd.DataFrame(self.detail)
            df_detail = df_detail[['open_time', 'open_price', 'volume', 'side', 'close_time', 'close_price', 'profit']]
            df_detail.to_csv(file_name, index = False)
        else:
            print('No history details to save')