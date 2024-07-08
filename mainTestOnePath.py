import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data_Generate import Data_Generate
from option_calculate import option_calculate
import json
import seaborn as sns
from Deep_model import LSTM_Hedging, GRU_Hedging
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
sns.set(style='darkgrid')
from Utils import raw_data_process

if __name__ == '__main__':
    data_path = 'CSI500.csv'
    option_path = 'options_attributes.json'
    data = pd.read_csv(data_path)
    data = raw_data_process(data)
    with open(option_path, 'r') as file:
        options = json.load(file)
    data_gene = Data_Generate(data)
    torch.set_default_tensor_type(torch.DoubleTensor)
    PATH0 = "DeepHedgingModel0.pt"
    DeepHedgingModel = torch.load(PATH0)
    test_size, simu_size = 1, 50000
    MC_Delta = []
    return_path = data_gene.garch_simulation(test_size, 306)
    option_cal = option_calculate(options, return_path, 15)
    price_path = np.cumprod(1 + return_path[:, 15:], axis=1)
    option_state, PnL = option_cal.state_observe(None, None, 0, np.ones(test_size))
    tensor_data1 = option_cal.tensor_transform(15, 15, 30, 1 / 30)
    tensor_data1 = torch.from_numpy(tensor_data1)
    option_PnL1 = torch.from_numpy(PnL)
    DeepDelta, PnL1 = DeepHedgingModel.forward(tensor_data1, option_PnL1, 0)
    DeepDelta = list(DeepDelta.flatten().detach().numpy())
    most_len = 291
    for day in tqdm(range(most_len)):
        simu_path = data_gene.garch_simulation(simu_size, most_len - day)
        options_his = option_cal.option_data_cut(None, day)
        price_now = price_path[:, day]
        delta = -option_cal.delta_clac(0.01, price_now, day, simu_path, options_his)
        MC_Delta.append(delta)
    fig = plt.figure()
    ax1 = fig.subplots()
    ax2 = ax1.twinx()  # 使用twinx()，得到与ax1 对称的ax2,共用一个x轴，y轴对称（坐标不对称）
    ax1.plot(list(price_path[0]), label='price')
    ax2.plot(MC_Delta, label='MC_Delta', c='g')
    ax2.plot(DeepDelta[1:], label='LSTM_Delta', c='r')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('price')
    ax2.set_ylabel('deltas')
    plt.legend()
    plt.show()
