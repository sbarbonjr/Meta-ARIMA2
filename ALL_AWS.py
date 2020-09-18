from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm_notebook as tqdm
from pandas import read_csv
import numpy as np
import pandas as pd
import tsfel
import os

# WandB – Import the wandb library
import wandb
wandb.init(project="meta-arima")

import warnings
warnings.filterwarnings("ignore")

# Avaliando o modelo ARIMA com ordens diferentes (p,d,q)
def evaluate_arima_model(X, arima_order):
  train_size = int(len(X) * 0.70)
  train, test = X[0:train_size], X[train_size:]
  history = [x for x in train]
  predictions = list()
  for t in range(len(test)):
    model = ARIMA(history, order=arima_order)
    model_fit = model.fit(disp=0)
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(test.iloc[t])
  rmse = sqrt(mean_squared_error(test, predictions))
  return rmse

def evaluate_models(dataset, p_values, d_values, q_values):
  dataset = dataset.astype('float32')
  best_score, best_cfg = float("inf"), None
  
  for p in p_values:
    for d in d_values:
      for q in q_values:
        order = (p,d,q)
        try:
          rmse = evaluate_arima_model(dataset, order) 
          wandb.log({"RMSE": rmse})
          if rmse < best_score:
            best_score, best_cfg = rmse, order
          #print('ARIMA%s RMSE=%.3f' % (order,rmse))
        except:
          continue
  #print('Melhor ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
  wandb.log({"Best RMSE": best_score})
  wandb.log({"Best p": best_cfg[0]})
  wandb.log({"Best d": best_cfg[1]})
  wandb.log({"Best q": best_cfg[2]})
  return best_cfg, best_score

file_ = "GAP_power_consumption"
url = "GAP_power_consumption.csv"    
series = pd.read_csv(url, sep=",", squeeze=True)
series = pd.to_numeric(series, errors='coerce')
series = pd.Series(np.nan_to_num(series))


#trocar para cada conta do COLAB!!! *******************************************
series = series[70000:]
output_file = "dataset_AWS_70k.csv"


if (os.path.exists(output_file)):
  df = pd.read_csv(output_file, index_col=False)
  atual = len(df)
else:
  atual=0

#Configuracao para GridSearch
p_values = [0, 1, 2, 4, 6, 8]
d_values = range(0, 3)
q_values = range(0, 3)

#p_values = [0]
#d_values = [0]
#q_values = [0]

cfg = tsfel.get_features_by_domain()
tam = 200
tam_janela = np.round(len(series)/tam)

wandb.log({"Instance":"AWS_70k"})
series_split = np.array_split(series, tam_janela)
dataset = pd.DataFrame()
cont = 0
for i in series_split[atual:]:
  cont = cont + 1
  result = adfuller(i)
  ADF = pd.DataFrame([result[0]])
  ADF_pvalue = pd.DataFrame([result[1]])
  if result[1] < 0.05:
    wandb.log({"Window":cont})
    wandb.log({"ADF_pvalue": result[1]})
    best_config, best_RMSE = evaluate_models(i, p_values, d_values, q_values)
    best_config = pd.DataFrame([best_config])
    best_config.columns = ["p","d","q"]
    #try:
    i = i.reset_index(drop=False)
    features = tsfel.time_series_features_extractor(cfg, i, verbose=10)
    features  = pd.concat([pd.DataFrame([file_]), pd.DataFrame([best_RMSE]), pd.DataFrame([tam]), ADF, ADF_pvalue, features, best_config], axis=1, ignore_index=False)
    #except:
      #print(i.values)
      #continue]
    features.to_csv(output_file, mode='a', header=False)
