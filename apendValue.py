import pandas as pd 

df = pd.read_csv('ComParE_2016_ad.csv',sep= ',', header = None)
print(df[6372])