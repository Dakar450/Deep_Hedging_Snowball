import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data_Generate import Data_Generate
from option_calculate import option_calculate
import json
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
sns.set(style='darkgrid')
from Utils import raw_data_process

data_path = 'CSI500.csv'
option_path = 'options_attributes.json'
data = pd.read_csv(data_path)
data = raw_data_process(data)
with open(option_path, 'r') as file:
    options = json.load(file)
data_gene = Data_Generate(data)
most_len, pre_len = 291, 15
batch_size, simu_size = 50, 10000
raw_PnL, Total_PnL = [], []

def MC_Hedge(i):
    delta0, delta, hedge_PnL = np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size)
    return_path = data_gene.garch_simulation(batch_size, most_len + pre_len)
    price_path = np.cumprod(1 + return_path, axis=1)
    option_cal = option_calculate(options, return_path, pre_len)
    option_state, PnL = option_cal.state_observe(None, None, 0, np.ones(batch_size))
    for day in tqdm(range(most_len)):
        simu_path = data_gene.garch_simulation(simu_size, most_len - day)
        options_his = option_cal.option_data_cut(None, day)
        price_now = price_path[:, day]
        delta = -option_cal.delta_clac(0.01, price_now, day, simu_path, options_his)
        hedge_PnL += (delta0 - delta) * price_now
        delta0 = delta
    hedge_PnL += delta * price_now
    sub_PnL = hedge_PnL + PnL
    return list(sub_PnL), list(PnL)


if __name__ == '__main__':
    pool = mp.Pool(3)
    res = pool.map_async(MC_Hedge, range(30))
    pool.close()
    pool.join()
    for item in res.get():
        raw_PnL.extend(item[1])
        Total_PnL.extend(item[0])
    raw_var = np.var(raw_PnL)
    hedged_var = np.var(Total_PnL)
    print("raw var: ", raw_var, "hedged_var: ", hedged_var)
    plt.hist(raw_PnL, bins=100, label="raw_PnL")
    plt.hist(Total_PnL, bins=50, alpha=0.5, label="hedged_PnL")
    plt.legend()
    plt.show()



