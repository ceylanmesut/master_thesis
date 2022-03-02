

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm 

import librosa
import librosa.display

import scipy.stats.mstats # for kurtosis

class Feature_Extractor:
    def __init__(self, path):

        self.y, self.sr = librosa.load(path, sr=44100)
        self.features = None

    def get_features(self, save = False):

        self._get_mfcc()
        self._get_mfcc_delta()
        self._get_mfcc_delta_delta()
        self._get_log_banks()
        self._get_log_banks_delta()
        self._get_log_banks_delta_delta()
        self._get_zero_crossing()
        self._get_kurtosis()

    def _concat_features(self, feature):
        

        self.features = np.hstack(
            [self.features, feature]
            if self.features is not None else feature)

    def _get_mfcc(self):

        # 1. Mel-Frequency Cepstral Coefficients: (n X n_mfcc 20)
        self.mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc = 20)
        mfcc_mean = np.mean((self.mfcc).T, axis=0)
        self._concat_features(mfcc_mean) 

    def _get_mfcc_delta(self):

        # 2. Mel-Frequency Cepstral Coefficients Delta: (n X n_mfcc 20)
        mfcc_delta = librosa.feature.delta(self.mfcc, mode = "constant")
        mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0) 
        self._concat_features(mfcc_delta_mean) 
    
    def _get_mfcc_delta_delta(self):

        # 3. Mel-Frequency Cepstral Coefficients Delta-Delta: (n X n_mfcc 20)
        mfcc_delta_delta = librosa.feature.delta(self.mfcc, mode = "constant", order = 2)
        mfcc_delta_delta_mean = np.mean(mfcc_delta_delta.T, axis=0) 
        self._concat_features(mfcc_delta_delta_mean)         

    def _get_log_banks(self):

        # 4. Log-Filterbanks (n X n_mels 20)
        specto = librosa.feature.melspectrogram(y = self.y, sr = self.sr, n_fft=400, hop_length=160, n_mels=20) 
        #Convert an amplitude spectrogram to dB-scaled spectrogram.
        self.log_specto = librosa.core.amplitude_to_db(specto)
        log_specto_mean = np.mean(self.log_specto, axis=1) #averaging out by samples and not mels >> returns 20 mels 
        #log_bank.append(log_specto_mean)
        self._concat_features(log_specto_mean)
        

    def _get_log_banks_delta(self):

        # 5. Log-Filterbanks Delta (n X n_mels 20)
        log_specto_delta = librosa.feature.delta(self.log_specto, mode = "constant")
        log_specto_delta_mean = np.mean(log_specto_delta.T, axis=0) 
        #log_bank_delta.append(log_specto_delta_mean)
        self._concat_features(log_specto_delta_mean)

    def _get_log_banks_delta_delta(self):

        # 6. Log-Filterbanks Delta-Delta (n X n_mels 20)
        log_specto_delta_delta = librosa.feature.delta(self.log_specto, mode = "constant", order = 2)
        log_specto_delta_delta_mean = np.mean(log_specto_delta_delta.T, axis=0) 
        #log_bank_delta2.append(log_specto_delta_delta_mean)
        self._concat_features(log_specto_delta_delta_mean)

    def _get_zero_crossing(self):
        
        # 7. Zero-Crossing Rate (n x 1)
        zero_rate = librosa.feature.zero_crossing_rate(self.y)
        zero_rate = np.mean(zero_rate, axis=1) # resulting 1 feature
        #zero_rates.append(zero_rate[0])
        self._concat_features(zero_rate)
        
    def _get_kurtosis(self):   

        # 8. Kurtosis (n x 1)
        kurtosis = scipy.stats.kurtosis(self.y)
        #kurtosis_list.append(kurtosis)
        self._concat_features(kurtosis)

# feature extraction main
if __name__ == "__main__":

    cough_file = pd.read_csv("main_data.csv")

    audio_features  = []

    # Defined for further partionining of data
    asthma = cough_file[cough_file["disease"]=="asthma"]# .loc[0:2000] #[0:20]
    copd = cough_file[cough_file["disease"]=="copd"]# .reset_index().loc[0:2000] #[0:20]
    covid = cough_file[cough_file["disease"]=="covid-19"]# .reset_index().loc[0:2000] #[0:20]

    subset = [asthma, copd, covid]

    for sub in subset:

         # Iterating the cough paths
        for i, row in tqdm(sub.iterrows(), total = sub.shape[0]):

            cough = Feature_Extractor(row["cough_path"]) # comes the y and sr from this
            cough.get_features()
            audio_features.append(cough)

    # returns n X 122 shape feature matrix
    feature_matrix = np.vstack([cough.features for cough in audio_features])
    
    with open("feature_matrix_new_laterversions.pkl", "wb") as file:
        pickle.dump(feature_matrix, file)
