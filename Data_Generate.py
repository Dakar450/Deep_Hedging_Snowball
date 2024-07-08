import numpy as np
import pandas as pd
from fitter import Fitter
from arch.univariate import ARX, GARCH
from tqdm import tqdm


class Data_Generate():

    def __init__(self, source_data):
        self.source_data = source_data
        f = Fitter(self.source_data['return'], distributions=['cauchy'])
        f.fit()
        self.cauchy_params = f.get_best(method='aic')['cauchy']
        ar = ARX(self.source_data['return'], lags=[1])
        ar.volatility = GARCH(p=1, o=0, q=1)
        self.GARCH_params = ar.fit().params

    def cauchy_simulation(self, num, length):
        params = self.cauchy_params
        data = np.random.standard_cauchy(size=num * length * 2)
        data = (data + params['loc']) * params['scale']
        data = data[(data > -0.09) & (data < 0.09)]
        data = data[0:num * length].reshape(num, length)
        return data

    def garch_simulation(self, num, length):
        params = self.GARCH_params
        data = self.source_data.copy()
        data['var'] = data['return'].rolling(15).var()
        data['epsilon'] = data['return']-params['Const']-params['return[1]']*data['return'].shift(1)
        data.dropna(inplace=True)
        data = data[['return', 'var', 'epsilon']]
        data_init = data.sample(n=num, replace=True)
        data_init.reset_index(drop=True, inplace=True)
        ret0, var0, epsilon0 = data_init['return'], data_init['var'], data_init['epsilon']
        path_generate = pd.DataFrame(ret0.copy())
        for i in range(length-1):
            var0 = params['omega']+params['alpha[1]']*epsilon0**2+params['beta[1]']*var0
            epsilon0 = np.sqrt(var0)*np.random.randn(num)
            ret0 = (ret0-params['Const'])*params['return[1]']+params['Const']+epsilon0
            path_generate[str(i)] = ret0
        data_generate = np.array(path_generate)
        return data_generate

    def bootstrap_simulation(self, num, length, boot_sub_len):
        data = self.source_data.copy()
        data['0'] = data['return']
        for i in range(boot_sub_len - 1):
            data[str(i + 1)] = data[str(i)].shift(-1)
        data = data[[str(i) for i in range(boot_sub_len)]].dropna()
        boot_num = int(np.ceil(num * length / boot_sub_len))
        data1 = data.sample(n=boot_num, replace=True)
        data1 = np.resize(np.array(data1), (num, length))
        return data1
