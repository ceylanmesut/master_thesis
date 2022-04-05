
#%%
from distutils.log import Log
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import GridSearchCV

from utils import Standardization
from dataset import Dataset
from trainer import Trainer
from flower_example import Federated_Learning
import fl_utils

# models = [LogisticRegression(solver = "newton-cg", penalty = "l2", C=1),
#           RandomForestClassifier(bootstrap = False, max_features="auto", n_estimators = 200),
#           GradientBoostingClassifier(max_depth = 5, n_estimators = 60),
#           DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_leaf=5)]

#models = [RandomForestClassifier(bootstrap=False, max_features="auto", n_estimators=5)]
#models = [LogisticRegression(solver="newton-cg", C=10, max_iter=5, penalty="l2")]

# models = [LogisticRegression(solver="newton-cg", C=1, penalty="l2")]
# params_lr = {'solver':('saga', 'newton-cg', 'sag', 'lbfgs'), 'penalty': ("l2", "none"), 'C':[0.01, 0.1, 1, 10]}

# Grid Search Models
#models = [LogisticRegression(), RandomForestClassifier(), GradientBoostingClassifier(), DecisionTreeClassifier()]

#params_lr = {'solver':('saga', 'newton-cg', 'sag', 'lbfgs'), 'penalty': ("l2", "none"), 'C':[0.01, 0.1, 1, 10]}
#params_rfc = {'bootstrap':[True, False], 'n_estimators':[10, 50, 100, 200, 500, 1000], 'max_features': ['auto', 'sqrt']}
#params_gbc = {'n_estimators':[20, 30, 40, 50, 60], 'max_depth':[3, 4, 5]}
#params_dtc = {'criterion': ['gini', 'entropy'], 'max_depth':[3,5,10,20], 'min_samples_leaf': [1,3,5]}


# hold_out = True
# grid_search = True

# if __name__ == "__main__":



###### Centralized Training ######   

# change feature matrix here    
# with open('feature_matrix_new_params_new_and_healthy_inc.pkl','rb') as f:
#     feature_matrix_loaded = pickle.load(f) 
#     print("Feature Matrix Shape:", feature_matrix_loaded.shape)  

     
# # dataset and label partioning  
# dataset = Dataset(feature_matrix=feature_matrix_loaded, cough_paths_csv="main_data_new_data_w-h.csv")
# X_train, y_train, X_test, y_test = dataset.get_train_test_data(ratio = 0.8)


# # training testing reporting
# trainer = Trainer(models, X_train, y_train, X_test, y_test, 
#                   save=True, 
#                   compute_shap=False, 
#                   do_grid_search=False, 
#                   grid_params=None
#                   )
# trainer.train_test_report()


#%% 
##### Federated Learning ######

with open('feature_matrix_new_params_new_and_healthy_inc.pkl','rb') as f:
    feature_matrix_loaded = pickle.load(f) 
    print("Feature Matrix Shape:", feature_matrix_loaded.shape)  
    
model = LogisticRegression(penalty="l2", max_iter=10, warm_start=True)
fl_utils.set_initial_params(model)


fed_learn = Federated_Learning(model, feature_matrix_loaded, num_clients=3)
fed_learn.simulate()



# %%
