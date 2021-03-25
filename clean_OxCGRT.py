import pandas as pd



df = pd.read_csv('Data/OxCGRT_latest.csv')

df.index.name = "Index"


#Dataframe without non-ordinal indicators and all the indices
df = df.drop(columns=['RegionName', 'RegionCode', 'E3_Fiscal measures', 'E4_International support', 'H4_Emergency investment in healthcare', 'H5_Investment in vaccines', 'M1_Wildcard', 'StringencyIndex', 'StringencyIndexForDisplay', 'StringencyLegacyIndex', 'StringencyLegacyIndexForDisplay', 'GovernmentResponseIndex', 'GovernmentResponseIndexForDisplay', 'ContainmentHealthIndex', 'ContainmentHealthIndexForDisplay', 'EconomicSupportIndex', 'EconomicSupportIndexForDisplay'])


df = df[df.Jurisdiction != "STATE_TOTAL"]

df = df.drop(columns=['Jurisdiction'])

df.to_csv('Data/OxCGRT_latest_cleaned.csv')

