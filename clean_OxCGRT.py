import pandas as pd



df = pd.read_csv('Data/OxCGRT_latest.csv')

df.index.name = "Index"

df = df.drop(columns=['RegionName', 'RegionCode', 'E3_Fiscal measures', 'E4_International support', 'H4_Emergency investment in healthcare', 'H5_Investment in vaccines', 'M1_Wildcard'])

df = df[df.Jurisdiction != "STATE_TOTAL"]

df = df.drop(columns=['Jurisdiction'])

df.to_csv('Data/OxCGRT_latest_cleaned.csv')

