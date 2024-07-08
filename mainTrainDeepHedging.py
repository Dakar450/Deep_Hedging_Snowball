import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data_Generate import Data_Generate
from option_calculate import option_calculate
import json
import seaborn as sns
import torch
from Deep_model import LSTM_Hedging, GRU_Hedging
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
sns.set(style='darkgrid')
from Utils import raw_data_process


def training_process(DeepModel, GeneModel, Option_data, period_epoch, max_period, batch_size, init_lr, init_loss, threshold, decay_ratio):
    mean_loss, lr = init_loss, init_lr
    loss_his = []
    optimizer = torch.optim.Adam(DeepModel.parameters(), lr=lr)
    for period in tqdm(range(max_period)):
        period_loss = []
        for epoch in tqdm(range(period_epoch)):
            return_path = GeneModel.garch_simulation(num = batch_size, length = 306)
            option_cal = option_calculate(Option_data, return_path, 15)
            option_state, PnL = option_cal.state_observe(None, None, 0, np.ones(batch_size))
            tensor_data = option_cal.tensor_transform(15, 15, 30, 1 / 30)
            tensor_data = torch.from_numpy(tensor_data)
            option_PnL = torch.from_numpy(PnL)
            delta_his, Total_PnL = Deep_Hedging_Model.forward(tensor_data, option_PnL, 0)
            loss = torch.var(Total_PnL)
            period_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm(Deep_Hedging_Model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
        loss_his.extend(period_loss)
        standard_loss_decre = (mean_loss-np.mean(period_loss))/(np.std(period_loss)/np.sqrt(period_epoch))
        mean_loss = np.mean(period_loss)
        if standard_loss_decre < threshold:
            lr = lr*decay_ratio
            optimizer.state_dict()['param_groups'][0]['lr'] = lr
        print("mean_loss: ", mean_loss, "standard_loss_decrement: ", standard_loss_decre, "lr: ", lr)
    return DeepModel, loss_his

if __name__ == '__main__':
    data_path = 'CSI500.csv'
    option_path = 'options_attributes.json'
    data = pd.read_csv(data_path)
    data = raw_data_process(data)
    with open(option_path, 'r') as file:
        options = json.load(file)
    data_gene = Data_Generate(data)
    batch_size = 1024
    torch.set_default_tensor_type(torch.DoubleTensor)
    Deep_Hedging_Model = LSTM_Hedging(22, 1, 20, 1)
    Deep_Hedging_Model, var_loss = training_process(Deep_Hedging_Model, data_gene, options, 150, 18, batch_size, 0.002, 0.012, 2, 0.3)
    plt.plot(var_loss)
    plt.show()
    PATH0 = "DeepHedgingModel0.pt"
    torch.save(Deep_Hedging_Model, PATH0)

