
#TODO
#from easydict import EasyDict as edict
import os
import matplotlib.pyplot  as plt
import pandas as pd
import time
from sklearn.metrics import classification_report, auc, accuracy_score
from sklearn.metrics import plot_confusion_matrix, precision_recall_fscore_support


class Trainer:
    def __init__(self, models, X_train, y_train, X_test, y_test, save=True):
        
        # models dict
        self.models = models

        # data
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test = y_test        

        # saving
        self.save = save
        
    def _train(self):

        self.trained_model = self.model.fit(self.X_train, self.y_train)

    def _predict(self):

        self.y_pred = self.model.predict(self.X_test)

    def _report(self):
        
        self.model_name = str(self.model).replace("'", "")
        self.model_name = self.model_name.replace(" ", "")

        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        print(classification_report(self.y_test, self.y_pred))
        print("Accuracy:%0.2f" % (self.accuracy))

        self.precision, self.recall, self.f_score, self.support = precision_recall_fscore_support(self.y_test, self.y_pred, average = "weighted")
        
        class_names = ['Asthma', 'COPD', 'COVID-19']
        plot_confusion_matrix(self.trained_model, self.X_test, self.y_test, normalize = "true", 
                                cmap=plt.cm.Blues, display_labels=class_names)
        
        plt.title("%s" % (self.trained_model), fontsize=10)
        
        # TODO change the naming issue here!!
        #plt.savefig("images\\conf_matrix_%s.png" % (self.model_name))
        plt.savefig("images\\conf_matrix_%s.png" % (time.strftime("%Y%m%d-%H%M%S")))
        #plt.close()
        #plt.show()        
        
    def _save_results(self):
        
        cols = ["model", "accuracy", "precision", "recall", "f_score"]
        if not os.path.isfile("results.xlsx"):
            
            # TODO check if file exists then no creation just continue writing
            # results = pd.DataFrame([{"precision": self.precision, "recall": self.recall, 
            #                         "f_score": self.f_score, "support": self.support}])
            #results = results.round(3)
            
            results = pd.DataFrame(columns = cols)        
            results.to_excel("results.xlsx", index=False)
        
        results = pd.read_excel("results.xlsx")
        line = [self.model_name, self.accuracy, self.precision, self.recall, self.f_score]
        results = results.append(pd.DataFrame([line], columns = cols), ignore_index=False)
        results = results.round(3)
        results.to_excel("results.xlsx", index=False)

    def train_test_report(self):
        
        for model in self.models:
            
            self.model = model
            print("Model:", self.model)
            print("------ Training is started! ------")
            self._train()
            self._predict()
            print("------ Training is completed! ------")
            print("------ Model Performance -----")
            self._report()
            
            if self.save:
                
                self._save_results()
                
                





