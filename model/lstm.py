import sys
sys.path.append('model/')
import pandas as pd
import torch.optim as optim
import torch.cuda as cuda
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt 
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def grab_last_batch(hn,dimension,hidden_size):
    return hn[:,-1,:].reshape(dimension,1,hidden_size)

def feature_label_split(df: pd.DataFrame, target_cols):
    y = df[target_cols]
    x = df.drop(columns=target_cols)
    return x, y

def get_mae_rmse(targets, predictions):
    """
    Tager targets og predictions for de 3 hubs og returnerer gennemsnitlig MAE og RMSE.
    """
    mean_abs_error = np.mean(np.abs(targets - predictions), axis=1)
    root_mean_squared_error = np.sqrt(np.mean(np.square(targets - predictions), axis=1))
    return np.mean(mean_abs_error), np.mean(root_mean_squared_error)

def calculate_excess_rows(df, sequence_length, batch_size):
    number_of_batches = len(df) // (sequence_length * batch_size)
    rows_to_drop = len(df) - (sequence_length * batch_size * number_of_batches)
    return rows_to_drop, number_of_batches

def reshape_dataframe(df, sequence_length, batch_size):
    rows_to_drop, number_of_batches = calculate_excess_rows(df, sequence_length, batch_size)
    df = df.iloc[rows_to_drop:, :]
    data = np.array(df.values)    
    data = data.reshape(number_of_batches, batch_size, sequence_length, df.shape[1])
    return data

def remove_outliers(df):
    df = df.apply(lambda x: [np.nan if y > 80 or y < -10 else y for y in x])
    df.interpolate(method='spline', order=1, limit=10, limit_direction='both',inplace=True)
    return df

def normalize_dataframe(df):
    return df.apply(lambda x: x-x.mean())

def differenciate_dataframe(df):
    return df.diff()

def lstm_train_test_splitter(   df_features: pd.DataFrame, 
                                df_targets: pd.DataFrame, 
                                sequence_length_train: int, 
                                batch_size_train: int, 
                                sequence_length_test: int = 24, 
                                batch_size_test: int = 1):
    """
    Splits the data into train and test sets. The test set will be the last
    test_size % of the data. The train set will be the remaining data.
    """    

    test_features = df_features.tail(sequence_length_test * batch_size_test)
    test_targets = df_targets.tail(sequence_length_test * batch_size_test)
    train_features = df_features.head(len(df_features) - len(test_features))
    train_targets = df_targets.head(len(df_targets) - len(test_features))

    x_train = reshape_dataframe(train_features, sequence_length_train, batch_size_train)
    y_train = reshape_dataframe(train_targets, sequence_length_train, batch_size_train)
    x_test = reshape_dataframe(test_features, sequence_length_test, batch_size_test)
    y_test = reshape_dataframe(test_targets, sequence_length_test, batch_size_test)
    return(x_train, y_train, x_test, y_test)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()
        self.output_dim = output_dim

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, dropout=dropout_prob, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)  

        #nn.init.xavier_uniform_(self.fc.weight)    <- xavier init
        #torch.randn instead of torch.zeros below to change to <- random init
    
    def create_hidden_states(self, batch_size):
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=device).requires_grad_()
        return (h0, c0)

    def forward(self, x, h0=None, c0=None):
        """x format is (batch_size, seq_len, input_dim)
        h0 and c0 can be inputted, they have the size (layer_dim, batch_size, hidden_dim) 
        where layer_dim is the number of stacked lstm's and hidden_dim is the number of nodes in each lstm"""
        if h0 is None or c0 is None:
            h0,c0 = self.create_hidden_states(x.size(0))

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out)
        
        if self.output_dim == 1:
            return (out[:,-1,-1],hn,cn)
        else:
            return (out,hn,cn)

class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []

    def train(self, train_features: torch.Tensor,train_targets:torch.Tensor, validation_features:torch.Tensor, validation_targets:torch.Tensor, n_epochs:int = 100000, model_statedict_path:str = "lstm_model.pth", forward_hn_cn:bool = False, plot_losses:bool = False):
        """
        hn,cn = opt.train(train_features=x_train,train_targets=y_train, validation_features=x_test, validation_target=y_test, n_epochs=n_epochs,forward_hn_cn=True,plot_losses=True, model_path = "lstm_model.pt")
        Trains the model and saves it to the model_path. The last hidden and cell states are returned.
        if forward_hn_cn is True, the last hidden and cell states are forwarded to the next batch. This is useful for training on sequences, with batchsize of 1.
        """

        # Early stopping
        best_val_loss = 9999999999999999999.9
        patience = 10 #bare til hyper tuning.
        trigger_times = 0
        epoch = 0

        #Load model again efter early stopping
        self.model.load_state_dict(torch.load(model_statedict_path))

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            h0,c0,hn,cn = None,None,None,None

            for train_batch, target_batch in zip(train_features, train_targets):
                loss,hn,cn = self.train_step(train_batch, target_batch, h0, c0)
                if forward_hn_cn:
                    h0 = hn
                    c0 = cn
                batch_losses.append(loss)

            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            if (epoch <= n_epochs) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t"
                )

            if epoch % 3 == 0:
                print("Running Validation")
                layer_dim = self.model.layer_dim
                hidden_dim = self.model.hidden_dim

                predictions = self.evaluate(validation_features,grab_last_batch(hn,layer_dim,hidden_dim),grab_last_batch(cn,layer_dim,hidden_dim))
                loss = self.calculate_loss(predictions, validation_targets)
                loss = float(format(loss, '.4f'))

                if loss >= best_val_loss: #ergo: den nye val er dårligere eller ligeså god
                    trigger_times += 1
                    print('Validation loss:', loss, "best loss so far is:", best_val_loss)  
                    print('Trigger Times:', trigger_times)
                else:
                    best_val_loss = loss
                    trigger_times = 0
                    print('Validation loss:', loss, "best loss so far is:", best_val_loss)   
                    torch.save(self.model.state_dict(), model_statedict_path)

                if trigger_times >= patience:
                    print('Early stopping activated the model performed worse', patience, "times in a row")  
                    break             

        if plot_losses:
            self.plot_losses()
        
        return (hn,cn)

    def train_step(self, x, y,h0=None,c0=None):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat,hn,cn = self.model(x,h0,c0)

        # Computes loss
        #loss = self.loss_fn(y, yhat)
        loss = self.loss_fn(y.to(device), yhat.to(device))
        if not (loss > 0):
            print("loss is not > 0")
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Returns the loss
        return (loss.item(),hn,cn)

    def evaluate(self, test_features, h0=None,c0=None,model_statedict_path = None):
        """
        Evaluates the model, by predicting a number of results. 
        It's possible to input the last hidden and cell states of the training data, if the model is trained on sequences.
        """
        if model_statedict_path != None:
            self.model.load_state_dict(torch.load(model_statedict_path))

        if torch.cuda.is_available():
            test_features = test_features.to(device)
        else:
            test_features.cpu()
            

        with torch.no_grad():
            predictions = []
            for test_batch in test_features:
                self.model.eval()
                
                yhat,hn,cn = self.model(test_batch.to(device),h0.to(device),c0.to(device))
                
                predictions.append(yhat.cpu().detach().numpy()) # predictions.append(new_predictions.cpu().detach().numpy())

                
                
        return torch.Tensor(np.array(predictions))

    def calculate_loss(self, yhat, targets):
        """
        Calculates the loss between the predictions and the targets.
        """
        losses = []
        for yhat_batch, target_batch in zip(yhat, targets):
            loss = self.loss_fn(yhat_batch, target_batch)
            losses.append(loss.item())
        loss = np.mean(losses)    
        return loss

    def plot_losses(self):
        
        plt.plot(self.train_losses, label="Training loss")
        #plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()



class LSTM():
    def __init__(self) -> None:
        self.targets_cols = ['TH_NP15_GEN-APND','TH_SP15_GEN-APND', 'TH_ZP26_GEN-APND']
        self.validation_length = 39
        self.validation_batchsize = 1

    def train(self,df: pd.DataFrame, hyper_dict):
        """
        Trains the model on 7 days and returns the loss of the best model
        Expected hyper_dict parameters:
        batch_size
        hidden_size       
        optimizer
        sequence_length
        lr
        weight_decay
        dropout_rate
        LSTM_layers
        days_training_length
        """
        df_features,df_targets = feature_label_split(df, self.targets_cols)
        df_features = df_features.drop(columns=['hour'], axis=1)

        #df_targets = remove_outliers(df_targets)
        df_features = normalize_dataframe(df_features)

        number_of_train_hours = len(df_features)-self.validation_length
        if number_of_train_hours*hyper_dict.batch_size < hyper_dict.sequence_length:
            raise Exception("Dataset is too short for the encoder length and batch size. Dataset length:{number_of_train_hours} encoder length:{sequence_length} batch size:{batch_size}")

        model_state_dict_path = "model/checkpoint/lstm_state_dict.pth"
        model_params = {'input_dim': len(df_features.columns), 'hidden_dim' : hyper_dict.hidden_size, 'layer_dim' : hyper_dict.LSTM_layers, 'output_dim' : 3, 'dropout_prob' : hyper_dict.dropout_rate}
        self.create_new_model(model_params)
        torch.save(self.model.state_dict(), model_state_dict_path)

        predictions = []
        targets = []
        MAEs = []
        RMSEs = []
        for i in range(7):
            print(f"Training for day {i}")
            df_features_loop = self.offset_dataframe(df_features,i)
            df_targets_loop = self.offset_dataframe(df_targets,i)

            x_train, y_train, x_test, y_test = lstm_train_test_splitter(df_features_loop, df_targets_loop, hyper_dict.sequence_length, hyper_dict.batch_size, self.validation_length, self.validation_batchsize)
            x_train, y_train, x_test, y_test = self.convert_to_tensors(x_train, y_train, x_test, y_test)

            optimizer = self.get_optimizer(hyper_dict.optimizer,hyper_dict.lr,hyper_dict.weight_decay)
            loss_function = torch.nn.L1Loss()
            opt = Optimization(model=self.model, loss_fn=loss_function, optimizer=optimizer)

            if hyper_dict.batch_size > 1:
                forward_hidden_states = False
            else:
                forward_hidden_states = True

            hn,cn = opt.train(x_train,y_train,x_test,y_test,model_statedict_path=model_state_dict_path,forward_hn_cn=forward_hidden_states)
            
            new_predictions = opt.evaluate(x_test,grab_last_batch(hn,hyper_dict.LSTM_layers,hyper_dict.hidden_size),grab_last_batch(cn,hyper_dict.LSTM_layers,hyper_dict.hidden_size),model_state_dict_path)
            mae,rmse = get_mae_rmse(new_predictions,y_test)

            MAEs.append(mae)
            RMSEs.append(rmse)
            predictions.append(new_predictions)
            targets.append(y_test)
        return MAEs, RMSEs, predictions, targets


    def get_optimizer(self,optimizer_name,learning_rate,weight_decay):
        if optimizer_name == "rAdam":
            return optim.RAdam(self.model.parameters(),lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "adam":
            return optim.Adam(self.model.parameters(),lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return optim.SGD(self.model.parameters(),lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "ranger":
            raise Exception("Ranger optimizer is not supported yet")
            return None
        else:
            raise Exception("Optimizer not found")

    def create_new_model(self,model_params):
        self.model = LSTMModel(**model_params)
        if torch.cuda.is_available():
            self.model.cuda()

    def offset_dataframe(self,df,i) -> pd.DataFrame:
        later_validation_hours = (6-i)*24
        unused_train_hours = i*24

        df_copy = df.drop(df.tail(later_validation_hours).index) # drop last n rows
        df_copy = df_copy.drop(df_copy.head(unused_train_hours).index) # drop first n rows
        return df_copy

    def convert_to_tensors(self,x_train, y_train, x_test, y_test):
        x_train = torch.Tensor(x_train)
        y_train = torch.Tensor(y_train)
        x_test = torch.Tensor(x_test)
        y_test = torch.Tensor(y_test)

        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            x_test = x_test.cuda()
            #y_test = y_test.cuda()

        return x_train, y_train, x_test, y_test