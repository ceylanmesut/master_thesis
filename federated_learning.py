

#%%
import flwr as fl
import numpy as np
import warnings
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import fl_utils
import pickle

from typing import Tuple, List, Optional, Dict
from flwr.common import (EvaluateRes, Scalar)

#%%
# np.random.seed(42)
# random.seed(42)

def main():

    # parameters 
    NUM_CLIENTS = 3
    NUM_ROUNDS = 4
    SAVE_RESULTS = True
    MODE = "client" # client or server 
    
    exp_path = fl_utils.create_experiment(MODE)
    model = LogisticRegression(penalty="l2", max_iter=100, warm_start=True, random_state=42)

    # client side FL
    if MODE == "client":
        strategy=AggregateCustomMetricStrategy(
                fraction_fit=1,  # Sample 10% of available clients for training
                fraction_eval=1,  # Sample 5% of available clients for evaluation
                min_fit_clients=3,  # Never sample less than 10 clients for training
                min_eval_clients=3,  # Never sample less than 5 clients for evaluation
               min_available_clients=3)
    
    # server side FL
    else: 
        strategy=AggregateCustomMetricStrategy(
                fraction_fit=1,  # Sample 10% of available clients for training
                fraction_eval=1,  # Sample 5% of available clients for evaluation
                min_fit_clients=3,  # Never sample less than 10 clients for training
                min_eval_clients=3,  # Never sample less than 5 clients for evaluation
                min_available_clients=3,
                eval_fn=get_eval_fn(model))        
    
    fl_utils.set_initial_params(model)
    history = fl.simulation.start_simulation(client_fn=client_fn, num_clients=NUM_CLIENTS,
                                    num_rounds=NUM_ROUNDS, strategy=strategy)
    
    fl_utils.report(history, exp_path, NUM_ROUNDS, SAVE_RESULTS, MODE)
    

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_val, y_val) -> None:
        
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        
    # returns current local model parameters 
    def get_parameters(self):
        
        params = fl_utils.get_model_parameters(self.model)
        return params
    
    # receives model parameters from the server, 
    # train the model parameters on the local data, 
    # and return the (updated) model parameters to the server 
    def fit(self, parameters, config):
        
        fl_utils.set_model_params(self.model, parameters)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.x_train, self.y_train)
        
        return fl_utils.get_model_parameters(self.model), len(self.x_train), {}     
            
    # evaluate global model parameters on local validation set
    # and return results
    def evaluate(self, parameters, config):

        fl_utils.set_model_params(self.model, parameters)    
            
        pred = self.model.predict_proba(self.x_val)     
        loss = log_loss(self.y_val, pred, labels=[0,1,2,3])
        accuracy = self.model.score(self.x_val, self.y_val)

        return loss, len(self.x_val), {"accuracy": accuracy}  

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):    
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[EvaluateRes],
        failures: List[BaseException]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        
        if not results:
            return None
        
        np.savez(f"fl_results\\round-{rnd}-results.npz", results) 
                    
        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        
        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")
        loss_aggregated=0
        # Call aggregate_evaluate from base class (FedAvg)
        params, _ = super().aggregate_evaluate(rnd, results, failures)
        
        return params, {"accuracy": accuracy_aggregated}
        # return loss_aggregated, {'accuracy': accuracy_aggregated}
        

    # to enable the Flower framework to create clients when necessary, 
    # we need to implement a function called client_fn 
    # that creates a FlowerClient instance on demand.
def client_fn(cid: str) -> fl.client.Client:
    # TODO: cough paths and feature matrix is hard coded
    # TODO: number of clients also

    clients = 3
    
    model = LogisticRegression(
        penalty="l2",
        max_iter=100, # local epoch
        warm_start=True, random_state=42) # prevent refreshing weights when fitting

    fl_utils.set_initial_params(model)
    
    # loading data
    with open('feature_matrix_new_params_new_and_healthy_inc.pkl','rb') as f:
        feature_matrix_loaded = pickle.load(f)      
    
    cough_data = pd.read_csv("main_data_new_data_w-h.csv")    
    
    X_train, _, y_train, _  = fl_utils.load_data(cough_data, feature_matrix_loaded)
    index_dict, X_train_loc, X_val_loc = fl_utils.partion_data(X_train, y_train, clients)
    X_train, X_val, y_train, y_val = fl_utils.get_data(index_dict, cough_data, feature_matrix_loaded, X_train_loc, X_val_loc, random_choice=int(cid))
    
    return FlowerClient(model, X_train, y_train, X_val, y_val)  
        

################## Centralized Evaluation ##################

# takes global model parameter, evaluates on 
#def _get_eval_fn(self, model: LogisticRegression):
#The current server implementation calls evaluate 
# after parameter aggregation and before federated evaluation
def get_eval_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""
        
    # TODO: cough paths and feature matrix is hard coded
    # loading data
    with open('feature_matrix_new_params_new_and_healthy_inc.pkl','rb') as f:
        feature_matrix_loaded = pickle.load(f)  
    
    cough_data = pd.read_csv("main_data_new_data_w-h.csv")
    
    _, X_test, _, y_test  = fl_utils.load_data(cough_data, feature_matrix_loaded)    
        
    # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        # Update model with the latest parameters
        fl_utils.set_model_params(model, parameters)

        pred = model.predict_proba(X_test)
        loss = log_loss(y_test, pred, labels=[0,1,2,3])
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate
      
      
if __name__ == "__main__":

    main()



# %%
