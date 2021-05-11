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
	#Convert the csv into an appropriate timeseries

	#Update timeseries so that the first value is the first confirmed case
	fnz = first_nonzero_case_index(cases)

	num_of_days_of_data = 365
	
	ts = cases[fnz:fnz+num_of_days_of_data] 


	#2D numpy array (doesn't include date column within the array)
	return cases, fnz




def plot_cases(time, cases, country_name):
	plt.figure(figsize=(10, 6))
	#plt.plot(time_valid, x_valid_unscaled, 'r-')
	plt.plot(range(10))
	plt.axvspan(3, 3, color='red', alpha=0.5)
	plt.title("Confirmed COVID-19 cases in " + country_name)
	plt.xlabel("Time")
	plt.ylabel("Confirmed Cases")

#	plt.savefig('filepath')
	plt.show()

# plots of cases from (say) four countries with a timeline from 1 January 2020 to the present and highlight the time period of interest for each country.



def main():
	countries = ['United_Kingdom', 'United_States', 'Greece', 'New_Zealand']	
	cases, first_case = setup_dataset('Greece')
	print(cases)
	print(first_case)
	


main()
