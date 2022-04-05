

from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression
# import openml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
plt.style.use('seaborn')


# source: https://flower.dev/blog/2021-07-21-federated-scikit-learn-using-flower/

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model):
    
    """Returns the paramters of a Logistic Regression model"""
    if model.fit_intercept:
        params = (model.coef_, model.intercept_)
        # print(params)
    else:
        params = (model.coef_,)
        # print(params)
    # print("get_model_parameters")
    return params    

def set_model_params(model: LogisticRegression, params: LogRegParams) -> LogisticRegression:
    """Sets the parameters of a sklearn LogisticRegression model"""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
        
    return model

def set_initial_params(model: LogisticRegression):
    """
    Sets initial parameters as zeros
    """
    # n_classes = 4 # MNIST has 10 classes
    # n_features = 98 # Number of features in dataset
    # model.classes_ = np.array([i for i in range(4)])
    
    n_classes = 4 # MNIST has 10 classes
    n_features = 98 # Number of features in dataset
    model.classes_ = np.array([i for i in range(n_classes)])    

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def load_data(cough_data, feature_matrix):
    """
    loads data from cough paths and extracted feature matrix
    """
    
    cough_data_grouped = cough_data.groupby(["patient_ID"])["disease"].value_counts().reset_index(name='counts')
    cough_data_grouped_X = cough_data_grouped[["counts", "patient_ID", "disease"]]
    cough_data_grouped_y = cough_data_grouped[["disease", "patient_ID"]]
    
    # Global splitting
    # TODO:Split parameter
    X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(cough_data_grouped_X, cough_data_grouped_y, 
                                                                            test_size=0.2, stratify=cough_data_grouped[["disease"]], random_state=42)   

    
    patients_test = X_test_main[X_test_main.index.isin(X_test_main.index.values)]["patient_ID"].values
    cough_indices_test = cough_data[cough_data["patient_ID"].isin(patients_test)].index.values

    # saving indices of the test data
    name_test = "test.csv"
    if not os.path.exists(name_test):
        
        test_df = pd.DataFrame(cough_indices_test)
        test_df.to_csv(name_test, index=False)
            
    X_test = feature_matrix[cough_indices_test] 
    
    y_test_main = cough_data[cough_data["patient_ID"].isin(patients_test)]["disease"].values
    y_test_main = encode_labels(cough_data).transform(y_test_main)
    
    return(X_train_main, X_test, y_train_main["disease"].values, y_test_main)

def partion_data(X_train, y_train, clients):
    # TODO: wwe call this function in client fn
    # TODO: this function should give me a X_train_loc Y_train_loc + VALIDATION SETS
    
    
    # Local splitting
    # TODO: Split parameter
    X_train_loc, X_val_loc, y_train_loc, y_val_loc = train_test_split(X_train, y_train, 
                                                                            test_size=0.2, stratify=y_train, random_state=42)       

    # then here I need to partion the local training data into num of clients
    # patient wise

    num_of_clients = clients
    
    X_train_loc_asthma_parts = np.array_split(X_train_loc[X_train_loc["disease"]=="asthma"].index, num_of_clients)
    X_train_loc_copd_parts = np.array_split(X_train_loc[X_train_loc["disease"]=="copd"].index, num_of_clients)
    X_train_loc_covid_parts = np.array_split(X_train_loc[X_train_loc["disease"]=="covid-19"].index, num_of_clients)
    X_train_loc_healthy_parts = np.array_split(X_train_loc[X_train_loc["disease"]=="healthy"].index, num_of_clients)

    train_1 = list(X_train_loc_asthma_parts[0].values) + list(X_train_loc_copd_parts[0].values) + list(X_train_loc_covid_parts[0].values) + list(X_train_loc_healthy_parts[0].values)                                                
    train_2 = list(X_train_loc_asthma_parts[1].values) + list(X_train_loc_copd_parts[1].values) + list(X_train_loc_covid_parts[1].values) + list(X_train_loc_healthy_parts[1].values)                     
    train_3 = list(X_train_loc_asthma_parts[2].values) + list(X_train_loc_copd_parts[2].values) + list(X_train_loc_covid_parts[2].values) + list(X_train_loc_healthy_parts[2].values)
                                                    
    # disease groups VALIDATION 
    X_val_loc_asthma_parts = np.array_split(X_val_loc[X_val_loc["disease"]=="asthma"].index, num_of_clients)
    X_val_loc_copd_parts = np.array_split(X_val_loc[X_val_loc["disease"]=="copd"].index, num_of_clients)
    X_val_loc_covid_parts = np.array_split(X_val_loc[X_val_loc["disease"]=="covid-19"].index, num_of_clients)
    X_val_loc_healthy_parts = np.array_split(X_val_loc[X_val_loc["disease"]=="healthy"].index, num_of_clients)

    val_1 = list(X_val_loc_asthma_parts[0].values) + list(X_val_loc_copd_parts[0].values) + list(X_val_loc_covid_parts[0].values) + list(X_val_loc_healthy_parts[0].values)                                                
    val_2 = list(X_val_loc_asthma_parts[1].values) + list(X_val_loc_copd_parts[1].values) + list(X_val_loc_covid_parts[1].values) + list(X_val_loc_healthy_parts[1].values)                     
    val_3 = list(X_val_loc_asthma_parts[2].values) + list(X_val_loc_copd_parts[2].values) + list(X_val_loc_covid_parts[2].values) + list(X_val_loc_healthy_parts[2].values)
    
    # this is the data dictionary that has the train and val splits patient-wise
    index_dict = {}

    index_dict[0] = (train_1, val_1)
    index_dict[1] = (train_2, val_2)
    index_dict[2] = (train_3, val_3)
            
    return(index_dict, X_train_loc, X_val_loc)

def get_data(data_dict, cough_data, feature_matrix, X_train_loc, X_val_loc, random_choice):
    
    # TODO
    # random_choice = np.random.choice(num_clients, replace=False)
    
    # getting indices of the respective client
    train_data_ind = data_dict[random_choice][0]
    val_data_ind = data_dict[random_choice][1]   

    # patients as keys    
    patients_train = X_train_loc[X_train_loc.index.isin(train_data_ind)]["patient_ID"].values
    patients_val = X_val_loc[X_val_loc.index.isin(val_data_ind)]["patient_ID"].values
    
    # using patients as keys, finding cough indices for feature matrix
    cough_indices_train = cough_data[cough_data["patient_ID"].isin(patients_train)].index.values
    cough_indices_val = cough_data[cough_data["patient_ID"].isin(patients_val)].index.values
    
    name_tr = "training_%s.csv" % str(random_choice) 
    name_val = "val_%s.csv" % str(random_choice) 
    
    if not os.path.exists(name_tr):
        
        tr_df = pd.DataFrame(cough_indices_train)
        tr_df.to_csv(name_tr, index=False)
        
    if not os.path.exists(name_val):  
        
        val_df = pd.DataFrame(cough_indices_val) 
        val_df.to_csv(name_val, index=False)               
        
    
    X_train = feature_matrix[cough_indices_train] 
    X_val = feature_matrix[cough_indices_val]
    
    y_train = cough_data[cough_data["patient_ID"].isin(patients_train)]["disease"].values
    y_val = cough_data[cough_data["patient_ID"].isin(patients_val)]["disease"].values
    
    y_train = encode_labels(cough_data).transform(y_train)
    y_val = encode_labels(cough_data).transform(y_val)
    
    return(X_train, X_val, y_train, y_val)

def encode_labels(cough_data):
    
    main_labels = np.array(cough_data[["disease"]]).reshape((-1,))
    encoder = LabelEncoder()
    encoded_labels  = encoder.fit_transform(main_labels)
    
    return(encoder)   

def report(history, export_path, num_rounds, save, mode):
    # TODO: make the plot function responsive to the num communication round
    # TODO: cough_data is hard coded
    
    cough_data = pd.read_csv("main_data_new_data_w-h.csv")
    
    results_dict = get_results_data(export_path)
    
    if mode=="client":
        plot_dist_eval(history, export_path, num_rounds, save)
        plot_client_performance(results_dict, history, export_path, num_rounds, save)
        
    else:
        plot_central_eval(history, export_path, num_rounds, save)

    files = get_split_indices(export_path)
    # print(files)
    # input()
    file_names = get_file_names(files)
    # print(file_names)
    # input()    
    # Plotting Coughs per Client and per Datasets
    data_dict, main_train, main_val, main_test = get_client_disease_dist(export_path, cough_data, files, file_names)
    plot_client_cough_dist(export_path, main_train, main_val, main_test, save)
    
    # Plotting Patients per Client and per Datasets
    data_dict, main_train, main_val, main_test = get_client_patient_dist(export_path, cough_data, files, file_names)
    plot_client_patient_dist(export_path, main_train, main_val, main_test, save)          
    

def get_file_names(files):
    
    names = []
    for i in files:
        
        split = i.split(".")
        names.append(split[0])
            
    return(names)
    
def create_experiment(mode):
    
    if not os.path.exists("FL_EXPERIMENTS"):
        os.mkdir("FL_EXPERIMENTS")
    
    dt_now = datetime.now().strftime('%Y-%m-%d %H-%M')

    file_name = dt_now.split(" ")[0] + "_" + dt_now.split(" ")[1] + "_" + mode
    
    exp_path = "FL_EXPERIMENTS" + "\\" + file_name
    print(exp_path)
    # input()
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
        
    return(exp_path)             
        
################## VISUALIZATIONS ##################
def get_client_disease_dist(export_path, cough_data, files, file_names):
    
    data_dict={}
    
    num_clients = len(files)-1
    
    for file, name in zip(files, file_names):
        
        path = export_path + "\\" + file
        file = pd.read_csv(path)

        coughs = cough_data[cough_data.index.isin(list(file["0"]))]["disease"]
        pivot_table = coughs.value_counts().rename_axis('disease').reset_index(name='counts')

        cl = pivot_table["disease"].nunique() # gives me number of client labels to add
        if name == "test":
            client_no = "Test Set"
            client_col = [client_no for i in range(0, cl)]
          
        else:
            client_no = name[-1]
            client_col = "Client" + " " + client_no # * cl
            client_col = [client_col for i in range(0, cl)]
        pivot_table["client"] = client_col
        
        data_dict[name] = pivot_table
   
    tr_data_df = pd.DataFrame()
    val_data_df = pd.DataFrame()
    test_data_df = pd.DataFrame()
    
    # building training set
    for ind, file_name in enumerate(file_names[0:int(num_clients/2)]):
        
        if ind == 0: 
            main_train = pd.concat([tr_data_df, data_dict[file_name]])
        else:
            main_train = pd.concat([main_train, data_dict[file_name]]) 
            
    # building validation set
    for ind, file_name in enumerate(file_names[int(num_clients/2):-1]):
        
        if ind == 0: 
            main_val = pd.concat([val_data_df, data_dict[file_name]])
        else:
            main_val = pd.concat([main_val, data_dict[file_name]]) 

    main_test = pd.concat([test_data_df, data_dict[file_names[-1]]])        
                
    return(data_dict, main_train, main_val, main_test)


def plot_client_cough_dist(export_path, main_train, main_val, main_test, save):
    
    fig, axs = plt.subplots(1, 3, figsize=(24,12))
    
    clients = ["Client 0", "Client 1", "Client 2"]
    diseases = ["Asthma", "COPD", "COVID-19", "Healthy"]
    legend = diseases
    colors = ["steelblue", "seagreen", "indianred", "gold"]

    title0="Client Training Cough distribution"
    title1="Client Validation Cough distribution"
    title2= "Global Test Set Cough distribution"
    
    main_train.groupby(["client", "disease"])["counts"].sum().unstack().plot(kind='bar', stacked=True, ax=axs[0], legend=False,
                                                                             rot=0)
    axs[0].set_title(title0, fontsize=17)
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Number of Coughs", fontsize = 13)
    
    for i in range(4):
        axs[0].bar_label(axs[0].containers[i], fmt="%i", fontsize=13, label_type='center')

    main_val.groupby(["client", "disease"])["counts"].sum().unstack().plot(kind='bar', stacked=True, ax=axs[1], rot=0)
    axs[1].set_title(title1, fontsize=17)
    axs[1].set_xlabel("")

    for i in range(4):
        axs[1].bar_label(axs[1].containers[i], fmt="%i", fontsize=13, label_type='center')
    axs[1].legend(legend, loc='best', fontsize=14)
        
    main_test.groupby(["client", "disease"])["counts"].sum().unstack().plot(kind='bar', stacked=True, ax=axs[2], legend=False, rot=0)
    axs[2].set_title(title2, fontsize=17)
    for i in range(4):
        axs[2].bar_label(axs[2].containers[i], fmt="%i", fontsize=13, label_type='center')    
        
    axs[2].set_xlabel("")
        
    if save:
        fig_path2 = export_path + "\\"+ "client_cough_dist.png"
        fig.savefig(fig_path2, dpi=400)        


def get_client_patient_dist(export_path, cough_data, files, file_names):

    data_dict={}
    
    num_clients = len(files)-1
    
    for file, name in zip(files, file_names):
        
        path = export_path + "\\" + file
        file = pd.read_csv(path)        
        
        patients = cough_data[cough_data.index.isin(list(file["0"]))][["disease","patient_ID"]]
        pivot_table = patients.groupby("disease").patient_ID.nunique()
        pivot_table = pivot_table.to_frame()
        
        cl = len(pivot_table) # gives me number of client labels to add

        if name == "test":
            client_no = "Test Set"
            client_col = [client_no for i in range(0, cl)]
            
        else:
            client_no = name[-1]
            client_col = "Client" + " " + client_no #client_no * cl
            client_col = [client_col for i in range(0, cl)]
        
        pivot_table["client"] = client_col
        data_dict[name] = pivot_table
        

    tr_data_df = pd.DataFrame()
    val_data_df = pd.DataFrame()
    test_data_df = pd.DataFrame()    
    
    # building training set
    for ind, file_name in enumerate(file_names[0:int(num_clients/2)]):
        
        if ind == 0:
            main_train = pd.concat([tr_data_df, data_dict[file_name]])

        else:
            main_train = pd.concat([main_train, data_dict[file_name]]) 
            

    # building validation set
    for ind, file_name in enumerate(file_names[int(num_clients/2):-1]):
        
        if ind == 0: 
            main_val = pd.concat([val_data_df, data_dict[file_name]])
        else:
            main_val = pd.concat([main_val, data_dict[file_name]]) 

    main_test = pd.concat([test_data_df, data_dict[file_names[-1]]])        
                
    return(data_dict, main_train, main_val, main_test)    


def plot_client_patient_dist(export_path, main_train, main_val, main_test, save):
    
    fig, axs = plt.subplots(1, 3, figsize=(24,12))
    
    clients = ["Client 0", "Client 1", "Client 2"]
    diseases = ["Asthma", "COPD", "COVID-19", "Healthy"]
    legend = diseases
    colors = ["steelblue", "seagreen", "indianred", "gold"]
    
    title0="Client Training Patient distribution"
    title1="Client Validation Patient distribution"
    title2= "Global Test Set Patient distribution"
   
    main_train.groupby(["client", "disease"])["patient_ID"].sum().unstack().plot(kind='bar', stacked=True, ax=axs[0], legend=False, rot=0)
    axs[0].set_title(title0, fontsize=17)
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Number of Patients", fontsize = 13)
    for i in range(4):
        axs[0].bar_label(axs[0].containers[i], fmt="%i", fontsize=13, label_type='center')    
       
    main_val.groupby(["client", "disease"])["patient_ID"].sum().unstack().plot(kind='bar', stacked=True, ax=axs[1], rot=0)
    axs[1].set_title(title1, fontsize=17)
    axs[1].set_xlabel("")
    for i in range(4):
        axs[1].bar_label(axs[1].containers[i], fmt="%i", fontsize=13, label_type='center')
    axs[1].legend(legend, loc='best', fontsize=14)    
    
    main_test.groupby(["client", "disease"])["patient_ID"].sum().unstack().plot(kind='bar', stacked=True, ax=axs[2], legend=False, rot=0)
    axs[2].set_title(title2, fontsize=17)
    axs[2].set_xlabel("")   
    for i in range(4):
        axs[2].bar_label(axs[2].containers[i], fmt="%i", fontsize=13, label_type='center')      
    
    if save:
        fig_path2 = export_path + "\\"+ "client_patient_dist.png"
        fig.savefig(fig_path2, dpi=400)   

def plot_dist_eval(hist, export_path, num_rounds, save):
    
    ########## Plotting Distributed Evaluation ##########    
    num_rounds=num_rounds

    number_of_rounds =  list(range(0, num_rounds))
    losses =[]
    metric = []

    for i in hist.losses_distributed :
        losses.append(i[1])
        
    for i in hist.metrics_distributed["accuracy"]:
        metric.append(i[1])
        
    plt.figure(figsize=(24,12))
    plt.subplot(121)
    plt.plot(number_of_rounds, losses, color='tab:orange')
    plt.title("Distributed Evaluation: Cross Enthropy Loss", fontsize=18)
    plt.ylabel("Loss", fontsize=16)
    plt.xlabel("Number of Communication Rounds", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0, 1.6])
    
    plt.subplot(122)
    plt.plot(number_of_rounds, metric, color='tab:blue')
    plt.title("Distributed Evaluation: Weighted Accuracy", fontsize=18)
    plt.ylabel("Weighted Accuracy", fontsize=16)
    plt.xlabel("Number of Communication Rounds", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0.5, 1.2])
    
    if save:
        fig_path2 = export_path + "\\"+ "dist_eval_acc_loss.png"
        plt.savefig(fig_path2, dpi=400)    
    
    
def plot_central_eval(hist, export_path, num_rounds, save=False):
########## Plotting Centralized Evaluation ########## 
    num_rounds=num_rounds

    number_of_rounds =  list(range(0, num_rounds+1))
    losses =[]
    metric = []

    for i in hist.losses_centralized:
        losses.append(i[1])
        
    for i in hist.metrics_centralized["accuracy"]:
        metric.append(i[1])
        
    plt.figure(figsize=(24,12))
    plt.subplot(121)
    plt.plot(number_of_rounds, losses, color='tab:orange')
    plt.title("Server Evaluation: Cross Enthropy Loss", fontsize=18)
    plt.ylabel("Loss", fontsize=16)
    plt.xlabel("Number of Communication Rounds", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0, 1.6])
    
    plt.subplot(122)
    plt.plot(number_of_rounds, metric, color='tab:blue')
    plt.title("Server Evaluation: Accuracy", fontsize=18)
    plt.ylabel("Accuracy", fontsize=16)
    plt.xlabel("Number of Communication Rounds", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0.5, 1.2])
    
    if save:
        fig_path2 = export_path + "\\"+ "central_eval_acc_loss.png"
        plt.savefig(fig_path2, dpi=400)    
        

def get_split_indices(export_path):
    
    train_files = glob.glob("training_*.csv")
    val_files = glob.glob("val_*.csv")
    test_files = glob.glob("test.csv")
    
    ind_files = train_files + val_files + test_files
    
    for file in ind_files:
        shutil.copy(file, export_path)
    
    return(ind_files)
    

def get_npz_files(export_path):
    
    fl_result_path = export_path + "\\" + "fl_results"
    if not os.path.exists(fl_result_path):
        os.mkdir(fl_result_path)    
    
    file_pattern = "fl_results\*.npz"
    result_files = glob.glob(file_pattern)

    for file in tqdm(result_files):
        shutil.copy(file, fl_result_path)
        
    return(fl_result_path)       

# Extracting results information from local log files
def get_results_data(export_path):
# def get_results_data(file_pattern="fl_results\*.npz"):
    
    fl_result_path = get_npz_files(export_path)
    
    get_split_indices(export_path)
    
    file_search = fl_result_path + "\\*.npz"
    result_files = glob.glob(file_search)        

    result_dict = {}

    # defining clients based on len of val set
    result_dict[404]={}
    result_dict[404]["loss"] = []
    result_dict[404]["acc"] = []

    result_dict[2059]={}
    result_dict[2059]["loss"] = []
    result_dict[2059]["acc"] = []

    result_dict[996]={}
    result_dict[996]["loss"] = []
    result_dict[996]["acc"] = []

    for file in tqdm(result_files):
        # print(file)
        # input()
        info = np.load(file, allow_pickle=True)
            
        for client in info["arr_0"]:
            
            c_id = client[1].num_examples
            acc = client[1].metrics["accuracy"]
            loss = client[1].loss
            
            result_dict[c_id]["loss"].append(loss)
            result_dict[c_id]["acc"].append(acc)
            
    return(result_dict)

# Plotting local performance
def plot_client_performance(result_dict, hist, export_path, num_rounds, save):

    losses =[]
    metric = []

    for i in hist.losses_distributed :
        losses.append(i[1])
        
    for i in hist.metrics_distributed["accuracy"]:
        metric.append(i[1])
        
    num_rounds = num_rounds
    number_of_rounds =  list(range(0, num_rounds))

    plt.figure(figsize=(24,12))
    plt.subplot(121)

    plt.plot(number_of_rounds, result_dict[2059]["loss"])
    plt.plot(number_of_rounds, result_dict[996]["loss"])
    plt.plot(number_of_rounds, result_dict[404]["loss"])
    plt.plot(number_of_rounds, losses)

    plt.title("Client Evaluation on Local Val Set: Cross Enthropy Loss", fontsize=18)
    plt.ylabel("Loss", fontsize=16)
    plt.xlabel("Number of Communication Rounds", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['Client 0', 'Client 1', 'Client 2', 'Average'], loc='best', fontsize=16)
    plt.ylim([0, 1.6])
    
    plt.subplot(122)
    plt.plot(number_of_rounds, result_dict[2059]["acc"])
    plt.plot(number_of_rounds, result_dict[996]["acc"])
    plt.plot(number_of_rounds, result_dict[404]["acc"])
    plt.plot(number_of_rounds, metric)
    plt.title("Client Evaluation on Local Val Set: Accuracy", fontsize=18)
    plt.ylabel("Accuracy", fontsize=16)
    plt.xlabel("Number of Communication Rounds", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['Client 0', 'Client 1', 'Client 2', 'Average'], loc='upper right', fontsize=16)
    plt.ylim([0.5, 1.2])
    
    if save:
        fig_path2 = export_path + "\\"+ "client_eval_acc_loss.png"
        plt.savefig(fig_path2, dpi=400)        