
import pickle
# import matplotlib.pyplot  as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier 


from utils import Standardization
from dataset import Dataset
from trainer import Trainer


models = [LogisticRegression(solver = "saga", penalty = "l2", max_iter=10, C=1, random_state=42),
          LogisticRegression(solver = "liblinear", penalty = "l1", max_iter=10, C=1, random_state=42),
          LogisticRegression(solver = "saga", penalty = "elasticnet", l1_ratio = 0.5, max_iter=10, C=1, random_state=42)]

if __name__ == "__main__":

    # feature matrix
    with open('feature_matrix_new.pkl','rb') as f:
        feature_matrix_loaded = pickle.load(f)

    scaler = Standardization(feature_matrix_loaded)
    feature_matrix_scaled = scaler.scale_features()

    # dataset and labels 
    dataset = Dataset()
    X_train, y_train, X_test, y_test = dataset.get_train_test_data(ratio = 0.8)

    # training testing reporting
    trainer = Trainer(models, X_train, y_train, X_test, y_test)
    trainer.train_test_report()

    

