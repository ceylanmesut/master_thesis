
from sklearn import preprocessing


class Standardization:
    def __init__(self, matrix):

        self.feature_matrix = matrix

    def scale_features(self):

        scaler = preprocessing.MinMaxScaler()
        self_scaled_fm = scaler.fit_transform(self.feature_matrix)
         
        return(self_scaled_fm)


class Encode_Labels():
    
    def __init__(self, labels):
    
        #self.encoded_labels = None
        self.labels = labels.ravel()
    
    def encode_labels(self):
        
        encoder = preprocessing.LabelEncoder()
        encoded_labels  = encoder.fit_transform(self.labels)
        
        # printout for label mapping
        print("Original Labels:", encoder.classes_)
        print("Corresponding Mapping:", encoder.transform(['asthma', 'copd', 'covid-19', 'healthy']))

        return(encoded_labels)



