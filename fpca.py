import matplotlib.pyplot as plt
import matplotlib.patches as pt
import numpy as np
import pandas as pd
from math import isnan
from fdasrsf import fPCA, time_warping, fdawarp, fdahpca


countries=['Aruba', 'Afghanistan', 'Angola', 'Albania', 'Andorra', 'United_Arab_Emirates', 'Argentina', 'Australia', 'Austria', 'Azerbaijan', 'Burundi', 'Belgium', 'Benin', 'Burkina_Faso', 'Bangladesh', 'Bulgaria', 'Bahrain', 'Bahamas', 'Bosnia_and_Herzegovina', 'Belarus', 'Belize', 'Bermuda', 'Bolivia', 'Brazil', 'Barbados', 'Brunei', 'Bhutan', 'Botswana', 'Central_African_Republic', 'Canada', 'Switzerland', 'Chile', 'China', "Cote_d'Ivoire", 'Cameroon', 'Democratic_Republic_of_Congo', 'Congo', 'Colombia', 'Cape_Verde', 'Costa_Rica', 'Cuba', 'Cyprus', 'Czech_Republic', 'Germany', 'Djibouti', 'Dominica', 'Denmark', 'Dominican_Republic', 'Algeria', 'Ecuador', 'Egypt', 'Eritrea', 'Spain', 'Estonia', 'Ethiopia', 'Finland', 'Fiji', 'France', 'Faeroe_Islands', 'Gabon', 'United_Kingdom', 'Georgia', 'Ghana', 'Guinea', 'Gambia', 'Greece', 'Greenland', 'Guatemala', 'Guam', 'Guyana', 'Hong_Kong', 'Honduras', 'Croatia', 'Haiti', 'Hungary', 'Indonesia', 'India', 'Ireland', 'Iran', 'Iraq', 'Iceland', 'Israel', 'Italy', 'Jamaica', 'Jordan', 'Japan', 'Kazakhstan', 'Kenya', 'Kyrgyz_Republic', 'Cambodia', 'South_Korea', 'Kuwait', 'Laos', 'Lebanon', 'Liberia', 'Libya', 'Liechtenstein', 'Sri_Lanka', 'Lithuania', 'Luxembourg', 'Latvia', 'Macao', 'Morocco', 'Monaco', 'Moldova', 'Madagascar', 'Mexico', 'Mali', 'Malta', 'Myanmar', 'Mongolia', 'Mozambique', 'Mauritania', 'Mauritius', 'Malaysia', 'Namibia', 'Niger', 'Nigeria', 'Nicaragua', 'Netherlands', 'Norway', 'Nepal', 'New_Zealand', 'Oman', 'Pakistan', 'Panama', 'Peru', 'Philippines', 'Papua_New_Guinea', 'Poland', 'Puerto_Rico', 'Portugal', 'Paraguay', 'Palestine', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saudi_Arabia', 'Sudan', 'Senegal', 'Singapore', 'Sierra_Leone', 'El_Salvador', 'San_Marino', 'Somalia', 'Serbia', 'Suriname', 'Slovak_Republic', 'Slovenia', 'Sweden', 'Eswatini', 'Seychelles', 'Syria', 'Chad', 'Togo', 'Thailand', 'Trinidad_and_Tobago', 'Tunisia', 'Turkey', 'Taiwan', 'Tanzania', 'Uganda', 'Ukraine', 'Uruguay', 'United_States', 'Uzbekistan', 'Venezuela', 'United_States_Virgin_Islands', 'Vietnam', 'South_Africa', 'Zambia', 'Zimbabwe']


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

	return ts


def plot_all_curves():
	fig, ax = plt.subplots(figsize=(10.0, 5.0))
	time = np.arange(365)

	for i, country in enumerate(countries):
		cases = setup_dataset(country)
		ax.scatter(time, cases) 

	plt.xlabel("Time")
	plt.ylabel("Confirmed Cases")
	plt.savefig('./cases_figs/all_countries_cases')
	plt.show()



# Save all cases as a npy file for quicker handling
def save_all_cases(countries):
	cases_all_countries = []
	for i, country in enumerate(countries):
		cases = setup_dataset(country)	
		cases_all_countries.append(cases)

	np.save('fpca', cases_all_countries)






def main():
	time = np.arange(365).astype('double')
	print(time.shape)
	#save_all_cases(countries)
	f = np.load('fpca.npy').astype('double').T #Need data in the form (days, countries)
	
	warp_f = time_warping.fdawarp(f, time)
	warp_f.srsf_align()

	#warp_f.plot()


	#FPCA
	#fPCA_analysis = fPCA.fdavpca(warp_f)
	fPCA_analysis = fPCA.fdajpca(warp_f)


	# Run the FPCA on a 3 components basis 
	num_components = 3
	fPCA_analysis.calc_fpca(no=num_components)
	#fPCA_analysis.plot()


	colors_legend = ['r', 'g', 'b']
	curves_legend = ["Curve 1", "Curve 2", "Curve 3"] 

	fig, ax = plt.subplots(figsize=(10.0, 5.0))
	for i in range(num_components):
		ax.scatter(time, fPCA_analysis.f_pca[:,0,i], c=colors_legend[i])
	

	legend_content = [pt.Patch(color=colors_legend[i], label=curves_legend[i]) for i in range(len(curves_legend))]
	plt.legend(handles=legend_content)
	plt.xlabel("Time")
	plt.ylabel("Confirmed Cases")
	#plt.savefig('./cases_figs/fpca_vertical')
	plt.savefig('./cases_figs/fpca_joint')
	plt.show()




main()







"""
dataset = skfda.datasets.fetch_growth()
fd = dataset['data']
y = dataset['target']
fd.plot()



fpca_discretized = FPCA(n_components=2)
fpca_discretized.fit(fd)
fpca_discretized.components_.plot()
"""
