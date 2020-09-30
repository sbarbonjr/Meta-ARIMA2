from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import pandas as pd
import tsfel
import os

from multiprocessing import Pool

# WandB – Import the wandb library
import wandb
import warnings
warnings.filterwarnings("ignore")

wandb.init(project="meta-arima")

N_CPUS = 1

# Global variable with results
results_dict = {}


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
    return rmse, arima_order


def evaluate_arima_callback(result):
    global results_dict
    rmse = result[0]
    order = result[1]

    wandb.log({"RMSE": rmse})
    results_dict[order] = rmse


def evaluate_models(dataset, p_values, d_values, q_values):
    global results_dict
    results_dict = {}

    dataset = dataset.astype('float32')

    pool = Pool(processes=N_CPUS)
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    pool.apply_async(evaluate_arima_model, (dataset, order),
                                     callback=evaluate_arima_callback)
                    # rmse = evaluate_arima_model(dataset, order)
                    # print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except Exception:
                    continue
    pool.close()
    pool.join()

    best_score, best_cfg = float("inf"), None

    for order, rmse in results_dict.items():
        if rmse < best_score:
            best_score, best_cfg = rmse, order

    # print('Melhor ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
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


# trocar para cada conta do COLAB!!! *******************************************
series = series[:100000]
output_file = "DatasetSignalAndTargets_0_100000.csv"


if (os.path.exists(output_file)):
    df = pd.read_csv(output_file, index_col=False)
    atual = len(df)
else:
    atual = 0

# Configuracao para GridSearch
p_values = [0, 1, 2, 4, 6, 8]
d_values = range(0, 3)
q_values = range(0, 3)

#p_values = [0]
#d_values = [0]
#q_values = [0]

cfg = tsfel.get_features_by_domain()
tam = 200
tam_janela = np.round(len(series)/tam)

series_split = np.array_split(series, tam_janela)
dataset = pd.DataFrame()
cont = 0
# wandb.log({"Instance": "ICMC"})
wandb.log({"Instance": "ExtractDatabase"})
for i in series_split[atual:]:
    result = adfuller(i)
    ADF = pd.DataFrame([result[0]])
    ADF_pvalue = pd.DataFrame([result[1]])
    if result[1] < 0.05:
        cont = cont + 1
        andamento = cont/len(series_split)*100
        wandb.log({"Window": cont})
        wandb.log({"Andamento":andamento})
        wandb.log({"ADF_pvalue": result[1]})
        best_config, best_RMSE = evaluate_models(i, p_values, d_values, q_values)
        best_config = pd.DataFrame([best_config])
        best_config.columns = ["p", "d", "q"]
        i = i.reset_index(drop=False)
        i = pd.DataFrame([i.iloc[:,1]])
        features = pd.concat([pd.DataFrame([file_]),  #nome do arquivo
                              pd.DataFrame([atual]),  #posicao da janela
                              pd.DataFrame([best_RMSE]), # RMSE da Config
                              pd.DataFrame([tam]),  #Tamanho da janela
                              ADF, #resultado do teste ADF
                              ADF_pvalue, #p-valor do ADF
                              best_config, # melhor ordem para a ARIMA
                              i], #todo o sinal
                              axis=1, ignore_index=False)
        features.to_csv(output_file, mode='a', header=False)