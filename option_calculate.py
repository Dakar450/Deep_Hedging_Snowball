import numpy as np
import copy

class option_calculate():

    def __init__(self, options_data, return_process, pre_days):
        self.options = options_data
        self.return_process = return_process
        # the exact pre_days day is included in the process if 0 is start
        self.pre_days = pre_days

    # day input here means the start of the day, so the day should be processed in this function
    def state_observe(self, option_state0, return_path, day, price_now):
        if option_state0 is None:
            option_state = copy.deepcopy(self.options)
        else:
            option_state = copy.deepcopy(option_state0)
        if return_path is None:
            return_path = copy.deepcopy(self.return_process)
            return_path = return_path[:, self.pre_days:]
        price_path = np.cumprod(1 + return_path, axis=1) * price_now[:, np.newaxis]
        option_PnL = np.zeros(return_path.shape[0])
        for option in option_state:
            if day <= option_state[option]['start_date']:
                option_state[option]['init_price'] = price_path[:, option_state[option]['start_date']]
                option_state[option]['knock_in_price'] = option_state[option]['knock_in'] * option_state[option]['init_price']
                option_state[option]['knock_out_price'] = option_state[option]['knock_out'] * option_state[option]['init_price']
                option_state[option]['knock_in_state'] = np.zeros((1, return_path.shape[0]), dtype = int)
                option_state[option]['knock_out_state'] = np.zeros((1, return_path.shape[0]), dtype = int)
                option_state[option]['knock_out_date'] = np.zeros((1, return_path.shape[0]), dtype = int)
                option_state[option]['observe_now'] = [i for i in option_state[option]['observe_date'] if i>=day]
            knock_out_date = option_state[option]['knock_out_date'][-1]
            for date in option_state[option]['observe_now']:
                date = date-day
                knock_in_his = option_state[option]['knock_in_state'][-1]
                knock_out_his = option_state[option]['knock_out_state'][-1]
                knock_in = (price_path[:, date]<option_state[option]['knock_in_price']).astype(int)
                knock_out = (price_path[:, date]>option_state[option]['knock_out_price']).astype(int)
                knock_in = knock_in_his+knock_in
                knock_in[knock_in>1] = 1
                knock_out = knock_out_his+knock_out
                knock_out[knock_out>1] = 1
                knock_out_incre = knock_out-knock_out_his
                knock_out_date_margin = date*knock_out_incre
                knock_in = knock_in-knock_out
                knock_in[knock_in<0] = 0
                knock_out_date += knock_out_date_margin
                option_state[option]['knock_in_state'] = np.concatenate((option_state[option]['knock_in_state'], knock_in.reshape(1, len(knock_in))), axis=0)
                option_state[option]['knock_out_state'] = np.concatenate((option_state[option]['knock_out_state'], knock_out.reshape(1, len(knock_out))), axis=0)
                option_state[option]['knock_out_date'] = np.concatenate((option_state[option]['knock_out_date'], knock_out_date.reshape(1, len(knock_out_date))), axis=0)
            if day<=option_state[option]['end_date']:
                knock_out_PnL = -(option_state[option]['knock_out_date'][-1]-option_state[option]['start_date'])*option_state[option]['rate']/250
                long_PnL = (price_path[:, option_state[option]["end_date"]-day]-option_state[option]['init_price'])/option_state[option]['init_price']
                knock_in_PnL = -long_PnL*option_state[option]['knock_in_state'][-1]
                non_knock = 1-option_state[option]['knock_out_state'][-1]-option_state[option]['knock_in_state'][-1]
                non_knock_PnL = -non_knock*option_state[option]['rate']*(option_state[option]['end_date']-option_state[option]['start_date'])/250
                knock_in_PnL[knock_in_PnL<0] = 0
                final_PnL = knock_out_PnL+knock_in_PnL+non_knock_PnL
                option_state[option]['final_PnL'] = final_PnL
                option_PnL += final_PnL*option_state[option]['weight']
        if option_state0 is None:
            self.options = option_state
        return option_state, option_PnL

    # day input here means the start of the day, so the data only contain the data before the end of the last day
    def option_data_cut(self, option_state, day):
        if option_state is None:
            option_state_part = copy.deepcopy(self.options)
        else:
            option_state_part = option_state
        for option in option_state_part:
            count = np.sum(np.array(option_state_part[option]['observe_date'])<day)+1
            option_state_part[option]['knock_in_state'] = option_state_part[option]['knock_in_state'][:count]
            option_state_part[option]['knock_out_state'] = option_state_part[option]['knock_out_state'][:count]
            option_state_part[option]['knock_out_date'] = option_state_part[option]['knock_out_date'][:count]
            option_state_part[option]['observe_now'] = option_state_part[option]['observe_date'][count-1:]
        return option_state_part

    # std_len should be less than pre_days-1
    # suggest return_scale = 15, std_scale = 30, day_scale = 1/30
    def tensor_transform(self, std_len, return_scale, std_scale, day_scale):
        return_path = self.return_process.copy()
        shift_size = std_len
        test_simu_3d = np.array([np.roll(return_path, i, axis=1) for i in range(shift_size)])
        test_simu_3d = test_simu_3d[:, :, self.pre_days:]
        state_matrix = return_path[:, self.pre_days:]
        batch_size, seq_size = state_matrix.shape
        std_simu = np.std(test_simu_3d, axis=0)
        price_path = np.cumprod(1 + return_path[:, self.pre_days:], axis=1).reshape(batch_size, seq_size, 1)
        state_matrix = state_matrix.reshape(batch_size, seq_size, 1)*return_scale
        state_matrix = np.concatenate((price_path, state_matrix, std_simu.reshape(batch_size, seq_size, 1)*std_scale), axis=2)
        options_data = self.options.copy()
        for option in options_data:
            start_date, end_date = options_data[option]['start_date'], options_data[option]['end_date']
            state = np.zeros((batch_size, seq_size, 1))
            state[:, start_date:end_date, :] = 1
            knock_in_price = price_path*options_data[option]['knock_in']
            knock_in_price[:, start_date:, :] = options_data[option]['knock_in_price'][:, None, None]
            knock_out_price = price_path*options_data[option]['knock_out']
            knock_out_price[:, start_date:, :] = options_data[option]['knock_out_price'][:, None, None]
            knock_in_state = np.array([]).reshape(batch_size, 0)
            knock_out_state = np.array([]).reshape(batch_size, 0)
            observe_date = np.array([options_data[option]['observe_date'][0]])
            day0, time = 0, 0
            for day in options_data[option]['observe_date']:
                knock_in = np.repeat(options_data[option]['knock_in_state'][time].reshape(batch_size, 1), day-day0, axis = 1)
                knock_out = np.repeat(options_data[option]['knock_out_state'][time].reshape(batch_size, 1), day-day0, axis =1 )
                knock_in_state = np.concatenate((knock_in_state, knock_in), axis = 1)
                knock_out_state = np.concatenate((knock_out_state, knock_out), axis = 1)
                observe_date = np.concatenate((observe_date, -1*(np.arange(day0, day)-day+1)), axis=0)
                day0 = day
                time += 1
            knock_in_state = np.concatenate((knock_in_state, options_data[option]['knock_in_state'][-1].reshape(batch_size, 1)), axis=1)
            knock_out_state = np.concatenate((knock_out_state, options_data[option]['knock_out_state'][-1].reshape(batch_size, 1)), axis=1)
            extra_time = seq_size-day-1
            knock_in_state = np.concatenate((knock_in_state, np.zeros((batch_size, extra_time))), axis = 1)
            knock_out_state = np.concatenate((knock_out_state, np.ones((batch_size, extra_time))), axis = 1)
            knock_in_state = knock_in_state.reshape((batch_size, seq_size, 1))
            knock_out_state = knock_out_state.reshape((batch_size, seq_size, 1))
            observe_date = np.concatenate((observe_date, np.ones(extra_time)*40), axis=0)
            observe_date = np.repeat(observe_date.reshape(1, seq_size), batch_size, axis=0)*day_scale
            observe_date = observe_date.reshape(batch_size, seq_size, 1)
            state_matrix = np.concatenate((state_matrix, state, observe_date, knock_in_price, knock_out_price, knock_in_state, knock_out_state), axis=2)
        return state_matrix

    # start of the day
    def value_calc(self, price_now, day, option_state, simu_path):
        option_state = copy.deepcopy(option_state)
        real_num, simu_num = price_now.shape[0], simu_path.shape[0]
        price_now_ = np.repeat(price_now, simu_num)
        return_path = np.tile(simu_path, (real_num, 1))
        return_path = np.insert(return_path, 0, 0, axis=1)
        for option in option_state:
            if day>=option_state[option]['start_date']:
                option_state[option]['init_price'] = np.repeat(option_state[option]['init_price'], simu_num)
                option_state[option]['knock_in_price'] = np.repeat(option_state[option]['knock_in_price'], simu_num)
                option_state[option]['knock_out_price'] = np.repeat(option_state[option]['knock_out_price'], simu_num)
                option_state[option]['knock_in_state'] = np.array([np.repeat(i, simu_num) for i in option_state[option]['knock_in_state']])
                option_state[option]['knock_out_state'] = np.array([np.repeat(i, simu_num) for i in option_state[option]['knock_out_state']])
                option_state[option]['knock_out_date'] = np.array([np.repeat(i, simu_num) for i in option_state[option]['knock_out_date']])
        options, PnL = self.state_observe(option_state, return_path, day, price_now_)
        options_value = np.mean(PnL.reshape(real_num, simu_num), axis=1)
        return options_value

    # start of the day
    def delta_clac(self, step, price_now, day, simu_path, option_state):
        value_up = self.value_calc(price_now+step*0.5, day, option_state, simu_path)
        value_down = self.value_calc(price_now-step*0.5, day, option_state, simu_path)
        delta = (value_up-value_down)/step
        return delta
