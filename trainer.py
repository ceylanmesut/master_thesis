
#TODO
#from easydict import EasyDict as edict
import os
import matplotlib.pyplot  as plt
import pandas as pd
import time
from sklearn.metrics import classification_report, auc, accuracy_score
from sklearn.metrics import plot_confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV

import shap # model interpretation

class Trainer:
    def __init__(self, models, X_train, y_train, X_test, y_test, save=True, compute_shap=False, do_grid_search = False, grid_params=None):
        
        # models dict
        self.models = models
        
        # grid search
        self.grid_search = do_grid_search # boolean
        self.grid_params = grid_params # dict

        # data
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test = y_test        

        # interpretability
        self.compute_shap=compute_shap
        # saving
        self.save = save
        
    def _train(self):
        
        if self.grid_search:
            self.clf = GridSearchCV(self.model, self.grid_params, cv = 5, scoring = "f1_weighted", verbose = 2)
            self.trained_model = self.clf.fit(self.X_train, self.y_train)
        else:  
            self.trained_model = self.model.fit(self.X_train, self.y_train)

    def _predict(self):

        if self.grid_search:
            self.y_pred = self.clf.best_estimator_.predict(self.X_test)
        else:
            self.y_pred = self.model.predict(self.X_test)

    def _report(self):
        
        if self.grid_search:
            print("-------GridSearch CV results-----")
            print("The best model: %s" % (self.trained_model.best_estimator_))
            print("The best score: %f" % (self.trained_model.best_score_))            
            self.trained_model = self.clf.best_estimator_

        
        self.model_name = str(self.model).replace("'", "")
        self.model_name = self.model_name.replace(" ", "")

        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        print(classification_report(self.y_test, self.y_pred))
        print("Accuracy:%0.2f" % (self.accuracy))

        self.precision, self.recall, self.f_score, self.support = precision_recall_fscore_support(self.y_test, self.y_pred, average = "weighted")
        
        # class_names = ['Asthma', 'COPD', 'COVID-19', 'Healthy']
        # plot_confusion_matrix(self.trained_model, self.X_test, self.y_test, normalize = "true", 
        #                         cmap=plt.cm.Blues, display_labels=class_names)
        
        # plt.title("%s" % (self.trained_model), fontsize=9)
        
        # # TODO change the naming issue here!!
        # #plt.savefig("images\\conf_matrix_%s.png" % (self.model_name))
        # plt.savefig("images\\conf_matrix_%s.png" % (time.strftime("%Y%m%d-%H%M%S")))

    def _save_results(self):
        
        
        if self.grid_search:
            
            grid_results = pd.DataFrame(self.clf.cv_results_, columns = self.clf.cv_results_.keys())
            grid_results.to_excel("%s_grid_search_results.xlsx" % (self.model_name))
            
        else:
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
    
    def _get_shap_values(self):
        
        # 1. Feature importance Plot by disease 
        fig1, ax1 = plt.subplots()
        explainer = shap.Explainer(self.model, self.X_test) 
        shap_val = explainer.shap_values(self.X_test)
    
        class_names = ['Asthma', 'COPD', 'COVID-19']
        shap.summary_plot(shap_val, self.X_test, plot_type="bar", class_names=class_names, show=False, max_display=122)
        
        # for saving 
        fig1.savefig("shap_feature_imp_bar2.pdf", bbox_inches='tight', dpi=300)      
        plt.close(fig1)  
        
        # 2. Beeswarm plot for feature impact on model
        fig2, ax2 = plt.subplots()
        explainer=shap.Explainer(self.model.predict, self.X_test)
        shap_val2= explainer(self.X_test)
        shap.plots.beeswarm(shap_val2, show=False, max_display=122)
        
        fig2.savefig("shap_feature_bee2.pdf", bbox_inches='tight', dpi=300)       
        plt.close(fig2)
        
        #explainer = shap.Explainer(self.model.predict, self.X_test)
        #s_values = explainer(self.X_test[0:10])
        #class_names = ['Asthma', 'COPD', 'COVID-19']
        #shap.summary_plot(s_values, self.X_test[0:10], plot_type="bar", class_names=class_names, show=True)       

        #print(type(s_values))
        #shap.waterfall_plot(explainer.base_values[0], s_values[0])
        #shap.plots.bar(s_values, show=True, max_display=30)
            
        # for saving 
        #plt.savefig("shap_bar2.pdf", bbox_inches='tight', dpi=300)
        
        #shap.summary_plot(s_values, plot_type='violin')
        #shap.plots.waterfall(s_values[0], max_display=10)
        #print(s_values)
        
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
            
            if self.compute_shap:
                self._get_shap_values()
            
            if self.save:
                
                self._save_results()
                

