import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


class Trading:

    def __init__(self, model_name, stoploss=10, max_dd=0.3, shuffle=False):
        self.model_name = model_name

        self.sc_X = MinMaxScaler(feature_range=(-1, 1))

        self.shuffle=shuffle
        self.data_length_1 = 48
        self.max_dd = max_dd
        self.cost_of_non_pos = 0.01
        self.takeprofit = 0.05
        self.stoploss = stoploss
        self.position = 0
        self.running_profit = 1.0
        self.start_bar = 0
        self.current_bar = self.start_bar
        self.steps_left = 0
        self.max_equity = 1.0
        self.min_after_max = 1.0
        self.max_drawdown = 0.0

        self.load_data(self.model_name)

    def load_data(self, filename):
        # filename = f"C:\\Users\\Administrator\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files\\data_for_ai_{filename}.csv"
        filename = f"data_for_ai_{filename}.csv"
        ds = pd.read_csv(
            filename,
            ";",
            index_col=False,
            low_memory=False
        )
        if self.shuffle:
            ds = ds.sample(frac=1)

        self.bar_datetime = ds.iloc[:, 0]
        self.move_down = ds.iloc[:, 5]
        self.move_up = ds.iloc[:, 6]
        self.y_base = ds.iloc[:, 7]
        self.y_base_3 = ds.iloc[:, 8]
        self.X = ds.iloc[:, 9:]
        self.X_scaled = self.sc_X.fit_transform(self.X).round(decimals=2)
        self.max_ = max(self.y_base)
        self.min_ = min(self.y_base)
        self.y_scaled = self.y_base / max(self.max_, abs(self.min_))
        self.y_forward = [np.sum(self.y_scaled[x:x + 48]) for x in range(len(self.y_scaled))]
        self.max_ = max(self.y_forward)
        self.min_ = min(self.y_forward)
        self.commission = 0.002
        self.commission_scaled = (self.commission) / (max(self.max_, abs(self.min_)))
        self.clip_scaled = (0.0001) / (max(self.max_, abs(self.min_)))


    def get_datetime(self, bar):
        return self.bar_datetime.iloc[bar]

    def get_state(self, bar, cur_position=0):
        cur_pos = np.array([cur_position])
        cur_pos = tf.convert_to_tensor(cur_pos)
        cur_pos = tf.expand_dims(cur_pos, 0)

        state_short = np.array(self.X_scaled[bar][:self.data_length_1])
        state_short = tf.convert_to_tensor(state_short)
        state_short = tf.expand_dims(state_short, 0)

        state_long = np.array(self.X_scaled[bar][self.data_length_1:])
        state_long = tf.convert_to_tensor(state_long)
        state_long = tf.expand_dims(state_long, 0)

        return cur_pos, state_short, state_long

    def get_reward(self, bar):
        return self.y_scaled[bar], self.y_base[bar]

    def reset(self, start_bar=0, steps_max=-1):
        self.position = 0
        self.running_profit = 1.0
        self.start_bar = start_bar
        self.current_bar = self.start_bar
        self.steps_left = steps_max
        self.max_equity = 1.0
        self.min_after_max = 1.0
        self.max_drawdown = 0.0
        return self.get_state(self.current_bar)

    def step(self, action, verbose=True):
        action = action - 1
        bar_start = self.get_datetime(self.current_bar)
        bar_end = ""  # self.get_datetime(self.current_bar+1)

        _, r_base = self.get_reward(self.current_bar)

        r_base *= action
        r_base = -self.cost_of_non_pos if action == 0 else r_base

        if self.position != action:
            self.position = action
            if action != 0:
                r_base -= self.commission

        r_scaled = 1 if r_base >= 0 else -1

        if action == 1 and self.move_down[self.current_bar] > self.stoploss:
            r_base = -self.stoploss
        if action == -1 and self.move_up[self.current_bar] > self.stoploss:
            r_base = -self.stoploss

        done = 0

        self.running_profit += r_base
        if self.running_profit > self.max_equity:
            self.max_equity = self.running_profit
            self.min_after_max = self.running_profit

        if self.min_after_max > self.running_profit:
            self.min_after_max = self.running_profit

        self.max_drawdown = max(self.max_drawdown, self.max_equity - self.min_after_max)

        if (self.max_equity != 0 and (1 - self.running_profit / self.max_equity) > self.max_dd) \
               or (self.max_equity == 0 and self.running_profit < -self.max_dd):
        #if (self.max_equity != 0 and (self.max_equity - self.running_profit) > self.max_dd) \
        #    or (self.max_equity == 0 and self.running_profit < -self.max_dd):
        # if self.max_drawdown >= self.max_dd:
            if verbose:
                print(f"{self.current_bar}: done due stoploss: {self.max_equity:.2f}->{self.running_profit:.2f}")
            done = 1

        self.current_bar += 1
        self.steps_left -= 1
        if (self.current_bar + 1 >= len(self.X) or self.steps_left == 0):
            done = 2

        return self.get_state(self.current_bar, self.position), r_scaled, r_base, done, bar_start, bar_end
