
#%%
import os
import glob
import pandas as pd
from tqdm import tqdm
import soundfile
import librosa
import numpy as np

#%%
############################## METHOD-2 ##############################

# Load the audio file
coswara_data_dir = os.path.abspath('Coswara-Data\\Extracted_data') # Local Path of iiscleap/Coswara-Data Repo
extracted_data_dir = os.path.join('Coswara-Data\\', 'Splitted_data2')  

if not os.path.exists(coswara_data_dir):
    raise("Check the Coswara dataset directory!")

if not os.path.exists(extracted_data_dir):
    os.makedirs(extracted_data_dir)

dirs_extracted = set(map(os.path.basename,glob.glob('{}/202*'.format(extracted_data_dir))))
dirs_all = set(map(os.path.basename,glob.glob('{}/202*'.format(coswara_data_dir))))

dirs_to_extract = list(set(dirs_all) - dirs_extracted)

dirs_to_extract = sorted(dirs_to_extract, key=lambda x: float(x))


for d in tqdm(dirs_to_extract):
    print(d)

    path = '{}\{}\*\cough-shallow.wav'.format(coswara_data_dir, d)
    
    files = glob.glob(path)

    for file in files:

        folder = file.split("\\")[-2]            
        export_path = "%s\%s\%s" % (extracted_data_dir, d, folder)
        
        if not os.path.exists(export_path):
            os.makedirs(export_path)        

        print(folder)

        
        try:
            y, sr = librosa.load(file, sr = 44100)
        except:
            print("File Length Zero")
            continue

        df = pd.DataFrame(y, columns = ["signal"])
        df["std"] = df.rolling(850).std()
        threshold = np.max(df["std"]) * 0.2
        df["threshold"] = threshold 

        # Finding start and end time stamps of the individual coughs
        starts = []
        ends = []

        for idx, _ in df.iterrows():
            
            if df["std"][idx] == None:
                continue

            if idx < (len(df) - 1):
                if df["std"][idx] < threshold and df["std"][idx+1] >= threshold:
                    starts.append(idx)
                    continue

                if df["std"][idx] >= threshold and df["std"][idx+1] < threshold:
                    ends.append(idx)
                    continue
            else:
                break     

        i = 0
        for start, end in zip(starts, ends):
            
            # slicing signal and saving
            y_new = np.array(df.loc[start:end, "signal"])
            soundfile.write("%s\\cough%i.wav" % (export_path, i), y_new, samplerate = sr)

            i += 1
