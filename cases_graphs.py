import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import isnan




#Finds the first non-zero confirmed case
def first_nonzero_case_index(cases):
	fnz = 0 
	for i in cases:
		if isnan(i) == False and i != 0:
			break
		fnz += 1	
	return fnz


def setup_dataset(country_name):
	df = pd.read_csv('Data/OxCGRT_latest_cleaned.csv', index_col='Index')

	df_country = df.loc[df['CountryName'] == country_name].fillna(0)
	cases = df_country['ConfirmedCases'].to_numpy()[:-1]
	
	fnz = first_nonzero_case_index(cases)

	num_of_days_of_data = 365
	
	ts = cases[fnz:fnz+num_of_days_of_data] 

	return cases, fnz




def plot_cases(time, cases, first_case, country_name):
	plt.figure(figsize=(10, 6))
	plt.plot(time, cases, 'b-')
	plt.axvspan(first_case, first_case+365, color='red', alpha=0.5)
	plt.title("Confirmed COVID-19 cases in " + country_name)
	plt.xlabel("Time")
	plt.ylabel("Confirmed Cases")
	plt.savefig('./cases_figs/cases_' + country_name)
	#plt.show()



def main():
	countries = ['United_Kingdom', 'United_States', 'Greece', 'New_Zealand']	
	for country in countries:
		cases, first_case = setup_dataset(country)
		plot_cases(range(len(cases)), cases, first_case, country)	




main()
