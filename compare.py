import os
import opensmile
import numpy as np
import glob
print("imported")
#from scipy.io import wavfile

# jtes angry path, 50 files
data_path ="/home/abhijith/Documents/Abijith dataset/ADReSSo21/diagnosis/train/audio/cn"
files = glob.glob(os.path.join(data_path, "*.wav"))
files.sort()

# initiate opensmile with emobase feature set
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)
smile.feature_names

# read wav files and extract emobase features on that file
feat = []

for file in files:
    print("processing file ... ", file)
    #sr, data = wavfile.read(file)
    #feat_i = smile.process_signal(data, sr)
    feat_i = smile.process_file(file)
    feat.append(feat_i.to_numpy().flatten())

# save feature as a csv file, per line, with comma
np.savetxt("ComParE_2016_cn.csv", feat, delimiter=",")   