
import pickle
import numpy as np
import pandas as pd
import random
from random import sample
random.seed(42)

from utils import Encode_Labels


class Dataset:

    def __init__(self):

        with open('feature_matrix_new.pkl','rb') as f:
            self.feature_matrix_loaded = pickle.load(f)        
        
        self.main_data = pd.read_csv("main_data.csv")

        self.cough_by_patient = self.main_data.groupby(["patient_ID"])["disease"].value_counts().reset_index(name='counts')
        self.copd_patients = self.cough_by_patient[self.cough_by_patient["disease"]=="copd"]
        self.covid_patients = self.cough_by_patient[self.cough_by_patient["disease"]=="covid-19"]
        self.asthma_patients = self.cough_by_patient[self.cough_by_patient["disease"]=="asthma"]

    def get_train_test_data(self, ratio = 0.8):
        
        self._explain_dataset()
        self._split_train_test(ratio)

        self.train_ind, self.test_ind = self._get_train_test_indices()

        labels = self._get_labels()
        label_encoder = Encode_Labels(labels)
        encoded_labels = label_encoder.encode_labels()
        
        X_train = np.delete(self.feature_matrix_loaded, self.test_ind, axis = 0)
        X_test = np.delete(self.feature_matrix_loaded, self.train_ind, axis = 0)

        y_train = np.delete(encoded_labels, self.test_ind, axis = 0)
        y_test = np.delete(encoded_labels, self.train_ind, axis = 0)        
             
        return(X_train, y_train, X_test, y_test)  
        
    def _explain_dataset(self):

        # returns the indices of the each patient group
        self.covid_p_index = self.covid_patients.index.tolist()
        self.copd_p_index = self.copd_patients.index.tolist()
        self.asthma_p_index = self.asthma_patients.index.tolist()

        # total patient per disease
        print("%i COVID-19 patients in all data." % len(self.covid_patients))
        print("%i COPD patients in all data." % len(self.copd_patients))
        print("%i Asthma  patients in all data." % len(self.asthma_patients))
        print("-------------------------------------------")

    def _split_train_test(self, ratio):
        
        # training & testing separation
        # training set patients
        self.covid_p_index_train = sample(self.covid_p_index, int(len(self.covid_p_index) * ratio)) 
        self.copd_p_index_train = sample(self.copd_p_index, int(len(self.copd_p_index) * ratio)) 
        self.asthma_p_index_train = sample(self.asthma_p_index, int(len(self.asthma_p_index) * ratio)) 

        # test set patients
        self.covid_p_index_test = np.setdiff1d(self.covid_p_index, self.covid_p_index_train)
        self.copd_p_index_test = np.setdiff1d(self.copd_p_index, self.copd_p_index_train)
        self.asthma_p_index_test = np.setdiff1d(self.asthma_p_index, self.asthma_p_index_train)

        # Training dataset patient amount
        print("%i COVID-19 patients in training data." % len(self.covid_p_index_train))
        print("%i COPD patients in training data." % len(self.copd_p_index_train))
        print("%i Asthma  patients in training data.\n" % len(self.asthma_p_index_train))
        
        # Test dataset patient amount
        print("%i COVID-19 patients in test data." % len(self.covid_p_index_test))
        print("%i COPD patients in test data." % len(self.copd_p_index_test))
        print("%i Asthma  patients in test data.\n" % len(self.asthma_p_index_test))         
        print("-------------------------------------------")
        # Training dataset patients and cough amounts
        total_covid_train = np.sum(self.covid_patients[self.covid_patients.index.isin(self.covid_p_index_train)])["counts"]
        total_copd_train = np.sum(self.copd_patients[self.copd_patients.index.isin(self.copd_p_index_train)])["counts"]
        total_asthma_train = np.sum(self.asthma_patients[self.asthma_patients.index.isin(self.asthma_p_index_train)])["counts"]

        print("%i COVID-19 coughs in training data." % (total_covid_train))
        print("%i COPD coughs in training data." % (total_copd_train))
        print("%i Asthma  coughs in training data." % (total_asthma_train))
        print("%i Total coughs in training set.\n" % (total_covid_train + total_copd_train + total_asthma_train))

        #print("COPD", self.copd_patients[self.copd_patients.index.isin(self.copd_p_index_test)])

        # Testing dataset patients and cough amounts
        total_covid_test = np.sum(self.covid_patients[self.covid_patients.index.isin(self.covid_p_index_test)])["counts"]
        total_copd_test = np.sum(self.copd_patients[self.copd_patients.index.isin(self.copd_p_index_test)])["counts"]
        total_asthma_test = np.sum(self.asthma_patients[self.asthma_patients.index.isin(self.asthma_p_index_test)])["counts"]

        print("%i COVID-19 coughs in test data." % (total_covid_test))
        print("%i COPD coughs in test data." % (total_copd_test))
        print("%i Asthma  coughs in test data." % (total_asthma_test))
        print("%i Total coughs in test set.\n" % (total_covid_test + total_copd_test + total_asthma_test))          
        print("-------------------------------------------")
        print("Data explanation is completed!")

    def _get_train_test_indices(self):

        train_covid = self.covid_patients[self.covid_patients.index.isin(self.covid_p_index_train)]["patient_ID"]
        train_copd = self.copd_patients[self.copd_patients.index.isin(self.copd_p_index_train)]["patient_ID"]
        train_asthma = self.asthma_patients[self.asthma_patients.index.isin(self.asthma_p_index_train)]["patient_ID"]

        test_covid = self.covid_patients[self.covid_patients.index.isin(self.covid_p_index_test)]["patient_ID"]
        test_copd = self.copd_patients[self.copd_patients.index.isin(self.copd_p_index_test)]["patient_ID"]
        test_asthma = self.asthma_patients[self.asthma_patients.index.isin(self.asthma_p_index_test)]["patient_ID"]

        # returning each observation belonging to the patients in training set per disease 
        train_covid_index = self.main_data[self.main_data["patient_ID"].isin(train_covid)].index
        train_copd_index = self.main_data[self.main_data["patient_ID"].isin(train_copd)].index
        train_asthma_index = self.main_data[self.main_data["patient_ID"].isin(train_asthma)].index

        # returning each observation belonging to the patients in test set per disease 
        test_covid_index = self.main_data[self.main_data["patient_ID"].isin(test_covid)].index
        test_copd_index = self.main_data[self.main_data["patient_ID"].isin(test_copd)].index
        test_asthma_index = self.main_data[self.main_data["patient_ID"].isin(test_asthma)].index

        # accumulating indices for training and test sets

        training_set_indices = train_covid_index.values.tolist() + train_copd_index.values.tolist() + train_asthma_index.values.tolist()  
        test_set_indices = test_covid_index.values.tolist() + test_copd_index.values.tolist() + test_asthma_index.values.tolist()  

        # sorted indices for the feature matrix
        training_set_indices = sorted(training_set_indices, reverse=False)
        test_set_indices = sorted(test_set_indices, reverse=False)

        return(training_set_indices, test_set_indices)
 

    def _get_labels(self):

        #asthma_lab = dataframe["disease"][dataframe["disease"]=="asthma"]#.reset_index()#.loc[0:2000].drop("index", axis=1)
        #copd_lab = dataframe["disease"][dataframe["disease"]=="copd"]#.reset_index()#.loc[0:2000].drop("index", axis=1)
        #covid_lab = dataframe["disease"][dataframe["disease"]=="covid-19"]#.reset_index()#.loc[0:2000].drop("index", axis=1)

        #labels = np.vstack((np.array(asthma_lab), np.array(copd_lab), np.array(covid_lab)))

        labels = self.main_data["disease"]

        return(labels)
        #return(asthma_lab, copd_lab, covid_lab)        