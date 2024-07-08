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
    test_size = 30000
    torch.set_default_tensor_type(torch.DoubleTensor)
    PATH0 = "DeepHedgingModel0.pt"
    DeepHedgingModel = torch.load(PATH0)
    return_path = data_gene.bootstrap_simulation(test_size, 306, 5)
    option_cal = option_calculate(options, return_path, 15)
    option_state, PnL = option_cal.state_observe(None, None, 0, np.ones(test_size))
    tensor_data = option_cal.tensor_transform(15, 15, 30, 1 / 30)
    tensor_data = torch.from_numpy(tensor_data)
    option_PnL = torch.from_numpy(PnL)
    delta_his, Total_PnL = DeepHedgingModel.forward(tensor_data, option_PnL, 0)
    Total_PnL = Total_PnL.flatten().detach().numpy()
    plt.hist(PnL, bins = 100, label = "raw_PnL")
    plt.hist(Total_PnL, bins = 50, alpha = 0.5, label = "DeepHedged_PnL")
    plt.legend()
    plt.show()
    print("variance without hedging: ", np.var(PnL), "variance after hedging: ", np.var(Total_PnL))


