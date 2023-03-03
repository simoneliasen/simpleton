import pandas as pd
import numpy as np
from datetime import datetime, timedelta
#import model.lstm_model
#from model.lstm import LSTM

from model.lstm import LSTM
import wandb
from dataclasses import dataclass

class Optimizers:
    rAdam = 'rAdam'
    adam = 'adam'
    sgd = 'sgd'
    ranger = 'ranger'

@dataclass
class Config:
    batch_size:int
    hidden_size:int
    attention_heads:int
    encoding_size:int
    optimizer:Optimizers
    encoder_length:int
    sequence_length:int
    lr:float
    weight_decay:float
    dropout_rate:float
    LSTM_layers:int
    n_encoder_layers:int
    n_decoder_layers:int
    days_training_length:int




def timestep_check(df_features, end_date):
    try:
        df_features[df_features['hour'] == str(end_date)].index[0]
        return True
    except:
        return False


def get_data(dates, traininglength, df_features):

    start_date = datetime.fromisoformat(dates)
    endindex = df_features.index[df_features['hour'] == dates][0]
    
    i = 7
    timestep_exist = False

    endate = start_date
    while not timestep_exist:
        endate -= timedelta(days=(traininglength+i))
        timestep_exist = timestep_check(df_features, endate)
        i += 1
        
    startindex = df_features.index[df_features['hour'] == str(endate)][0]
    df_season = df_features.iloc[startindex:endindex]
   
    return df_season


def run():
    df_features = pd.read_csv("data/dataset.csv")
    hyperparameters = Config(1, 461, None, None, "sgd", None, 32, 0.12467119874140932, 0.0019913732590904044, 0.0755042607191115, 3, None, None, 36)
    season =  ["Winter", "Spring", "Summer", "Fall"]
    dates = ["2021-01-10 23:00:00", "2021-04-11 23:00:00", "2021-07-11 23:00:00", "2021-10-10 23:00:00" ]
    Total_average_mae_loss = 0
    Total_average_rmse_loss = 0
    Mae_Season_list = []
    RMSE_Season_list = []
    Season_MAE_Loss = []
    Season_RMSE_Loss = []
    targets_season = []
    predictions_season = [[],[],[],[],[]] #5 ensembles.
    for x in range(len(dates)):
        for ensemble_index in range(5):
            df_season = get_data(dates[x], hyperparameters.days_training_length, df_features)

            lstm_obj = LSTM()
            mae, rmse, predictions, targets = lstm_obj.train(df_season, hyperparameters)
         
            Mae_Season_list.append(mae)
            RMSE_Season_list.append(rmse)  
            average_mae_season = np.mean(mae)
            average_rmse_season = np.mean(rmse)
            Season_MAE_Loss.append(average_mae_season)
            Season_RMSE_Loss.append(average_rmse_season)
            Total_average_mae_loss += average_mae_season
            Total_average_rmse_loss += average_rmse_season

            targetshub1= []
            targetshub2= []
            targetshub3= []
            predhub1 = []
            predhub2= []
            predhub3= []
            for z in range(len(predictions)):
                for y in range(len(predictions[z])):
                    for q in range(len(predictions[z][y])):                    
                        for d in range(len(predictions[z][y][q])):
                            for p in range(len(predictions[z][y][q][d])):
                                if p == 0:                                
                                    targetshub1.append(targets[z][y][q][d][p])
                                    predhub1.append(predictions[z][y][q][d][p])
                                if p == 1:
                                    targetshub2.append(targets[z][y][q][d][p])
                                    predhub2.append(predictions[z][y][q][d][p])
                                if p == 2:
                                    targetshub3.append(targets[z][y][q][d][p])
                                    predhub3.append(predictions[z][y][q][d][p])

            if ensemble_index == 0:
                targets_season.append(targetshub1) 
                targets_season.append(targetshub2)
                targets_season.append(targetshub3)
            predictions_season[ensemble_index].append(predhub1)
            predictions_season[ensemble_index].append(predhub2)
            predictions_season[ensemble_index].append(predhub3)

    notfirst15 = 14
    stepmove1 = 39
    stepmove2 = 39
    maes = []
    rmses = []

    for f in range(len(targets_season[0])):
        if f == stepmove2:
            notfirst15+=stepmove1
            stepmove2 += stepmove1
        if f > notfirst15:
            logs:dict = {}
            for ensemble_idx in range(len(predictions_season)):
                logs.update({f"Ensemble {ensemble_idx} - {season[0]} predictions (24h) Hub: NP15": predictions_season[ensemble_idx][0][f], f"Ensemble {ensemble_idx} - {season[0]} targets (24h) Hub: NP15": targets_season[0][f], f"Ensemble {ensemble_idx} - {season[0]} predictions (24h) Hub: SP15": predictions_season[ensemble_idx][1][f], f"Ensemble {ensemble_idx} - {season[0]} targets (24h) Hub: SP15": targets_season[1][f], f"Ensemble {ensemble_idx} - {season[0]} predictions (24h) Hub: ZP26": predictions_season[ensemble_idx][2][f], f"Ensemble {ensemble_idx} - {season[0]} targets (24h) Hub: ZP26": targets_season[2][f], f"Ensemble {ensemble_idx} - {season[1]} predictions (24h) Hub: NP15": predictions_season[ensemble_idx][3][f], f"Ensemble {ensemble_idx} - {season[1]} targets (24h) Hub: NP15": targets_season[3][f], f"Ensemble {ensemble_idx} - {season[1]} predictions (24h) Hub: SP15": predictions_season[ensemble_idx][4][f], f"Ensemble {ensemble_idx} - {season[1]} targets (24h) Hub: SP15": targets_season[4][f], f"Ensemble {ensemble_idx} - {season[1]} predictions (24h) Hub: ZP26": predictions_season[ensemble_idx][5][f], f"Ensemble {ensemble_idx} - {season[1]} targets (24h) Hub: ZP26": targets_season[5][f],f"Ensemble {ensemble_idx} - {season[2]} predictions (24h) Hub: NP15": predictions_season[ensemble_idx][6][f], f"Ensemble {ensemble_idx} - {season[2]} targets (24h) Hub: NP15": targets_season[6][f], f"Ensemble {ensemble_idx} - {season[2]} predictions (24h) Hub: SP15": predictions_season[ensemble_idx][7][f], f"Ensemble {ensemble_idx} - {season[2]} targets (24h) Hub: SP15": targets_season[7][f], f"Ensemble {ensemble_idx} - {season[2]} predictions (24h) Hub: ZP26": predictions_season[ensemble_idx][8][f], f"Ensemble {ensemble_idx} - {season[2]} targets (24h) Hub: ZP26": targets_season[8][f], f"Ensemble {ensemble_idx} - {season[3]} predictions (24h) Hub: NP15": predictions_season[ensemble_idx][9][f], f"Ensemble {ensemble_idx} - {season[3]} targets (24h) Hub: NP15": targets_season[9][f], f"Ensemble {ensemble_idx} - {season[3]} predictions (24h) Hub: SP15": predictions_season[ensemble_idx][10][f], f"Ensemble {ensemble_idx} - {season[3]} targets (24h) Hub: SP15": targets_season[10][f], f"Ensemble {ensemble_idx} - {season[3]} predictions (24h) Hub: ZP26": predictions_season[ensemble_idx][11][f], f"Ensemble {ensemble_idx} - {season[3]} targets (24h) Hub: ZP26": targets_season[11][f]})
            wandb.log(logs)

        preds_np15 = []
        preds_sp15 = []
        preds_zp15 = []
        target_np15 = 0
        target_sp15 = 0
        target_zp15 = 0

        for ensemble_idx in range(len(predictions_season)):
            preds_np15.append(predictions_season[ensemble_idx][0][f])
            preds_sp15.append(predictions_season[ensemble_idx][1][f])
            preds_zp15.append(predictions_season[ensemble_idx][2][f])
            target_np15 = targets_season[0][f]
            target_sp15 = targets_season[1][f]
            target_zp15 = targets_season[2][f]

        ae_sp15 = abs(target_sp15 - (sum(preds_sp15) / len(preds_sp15)))
        ae_np15 = abs(target_np15 - (sum(preds_np15) / len(preds_np15)))
        ae_zp15 = abs(target_zp15 - (sum(preds_zp15) / len(preds_zp15)))

        maes.append(ae_sp15)
        maes.append(ae_zp15)
        maes.append(ae_np15)

        rmses.append(ae_np15 * ae_np15)
        rmses.append(ae_sp15 * ae_sp15)
        rmses.append(ae_sp15 * ae_sp15)

    avg_mae = sum(maes) / len(maes)
    avg_rmse = sum(rmses) / len(rmses)

    print("avg_mae:", avg_mae)
    print("avg_rmse:", avg_rmse)
    wandb.log({"Total_Average_RMSE_Loss": avg_rmse, "Total_Average_MAE_Loss": avg_mae})


run()
