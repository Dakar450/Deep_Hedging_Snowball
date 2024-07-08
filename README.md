# Deep_Hedging
dynamic hedging snowball option sets by deep hedging

## Instruments
### Data_Generate.py
generate return processes by Geometric Browning Motion Process with cauchy distribution, AR(1)-GARCH(1, 1) process and bootstrap method
### option_calculate.py
calculate options state, value and delta
### Deep_model.py
seq2seq deep learning models, including LSTM and GRU
### Utils.py
data reading function

## Interfaces
### mainTrainDeepHedging.py
interface to train the deep hedging model
### mainMC_Hedging.py
interface to test dynamic hedging based on Monte Carlo Simulation
### mainTestDeepHedging.py
interface to test deep hedging model
