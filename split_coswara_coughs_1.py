

import os
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence

# source: Coswara dataset github and stackoverflow

############################## METHOD-1 ##############################
# Load the audio file
coswara_data_dir = os.path.abspath('Coswara-Data\\Extracted_data') # Local Path of iiscleap/Coswara-Data Repo
extracted_data_dir = os.path.join('Coswara-Data\\', 'Splitted_data')  

if not os.path.exists(coswara_data_dir):
    raise("Check the Coswara dataset directory!")

if not os.path.exists(extracted_data_dir):
    os.makedirs(extracted_data_dir)

dirs_extracted = set(map(os.path.basename,glob.glob('{}/202*'.format(extracted_data_dir))))
dirs_all = set(map(os.path.basename,glob.glob('{}/202*'.format(coswara_data_dir))))

dirs_to_extract = list(set(dirs_all) - dirs_extracted)

for d in tqdm(dirs_to_extract):

    path = '{}\{}\*\cough-shallow.wav'.format(coswara_data_dir, d)    
    files = glob.glob(path)

    for file in files:

        audio_file = AudioSegment.from_wav(file)

        # silence chunk:500 ms
        # silence threshold = -40 dBFS
        chunks = split_on_silence(audio_file, min_silence_len = 100, silence_thresh = -20)

        # Process each part with your parameters
        for i, split in enumerate(chunks):
            # silence chunk 200 ms
            silence_chunk = AudioSegment.silent(duration=200)

            # Padding audio chunk with silence part
            audio_chunk = silence_chunk + split + silence_chunk

            # TODO: should it be normalized ??
            # Normalize the entire chunk.
            #normalized_chunk = match_target_amplitude(audio_chunk, -20.0)

            folder = file.split("\\")[-2]            
            export_path = "%s\%s\%s" % (extracted_data_dir, d, folder)

            if not os.path.exists(export_path):
                os.makedirs(export_path)

            audio_chunk.export(("%s\\cough%i.wav" % (export_path, i)), bitrate = "192k", format = "wav")
            

# min_silence_len = 500, silence_thresh = -40
#3     875
# 1     597
# 2     367
# 4     282
# 5      74
# 6      28
# 7       9
# 8       7
# 10      5
# 9       1
# 13      1
# 17      1

# min_silence_len = 400, silence_thresh = -40
# 3     883
# 1     458
# 4     358
# 2     353
# 5     110
# 6      48
# 7      17
# 8       9
# 10      3
# 9       3
# 11      3
# 14      1
# 18      1
# 15      1

# min_silence_len = 300, silence_thresh = -40
# 3     830
# 4     421
# 1     341
# 2     302
# 5     186
# 6      82
# 7      42
# 8      18
# 9      15
# 11      7
# 10      4
# 14      1
# 18      1
# 22      1
# 15      1
# 17      1

# min_silence_len = 300, silence_thresh = -20)
# 1     503
# 3     417
# 2     332
# 4     128
# 5      47
# 6      17
# 7       9
# 9       4
# 8       3
# 11      2
# 10      1
# 12      1
# 17      1
# Name: counts, dtype: int64

# min_silence_len = 300, silence_thresh = -30)
# 3     811
# 4     348
# 1     342
# 2     312
# 5     128
# 6      69
# 7      38
# 8      23
# 9       7
# 10      6
# 12      3
# 13      3
# 11      2
# 20      1
# 26      1
# 15      1
# 21      1

#min_silence_len = 100, silence_thresh = -30)
# 3     525
# 4     324
# 5     208
# 6     206
# 9     181
# 7     160
# 8     141
# 2     118
# 1     111
# 10     79
# 11     46
# 12     30
# 13     17
# 15     11
# 14      6
# 17      5
# 18      5
# 19      3
# 23      2
# 16      2
# 40      1
# 38      1
# 22      1
# 20      1
# 33      1