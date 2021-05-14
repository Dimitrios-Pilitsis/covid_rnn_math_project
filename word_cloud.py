import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import isnan
from wordcloud import WordCloud




countries=['Aruba', 'Afghanistan', 'Angola', 'Albania', 'Andorra', 'United_Arab_Emirates', 'Argentina', 'Australia', 'Austria', 'Azerbaijan', 'Burundi', 'Belgium', 'Benin', 'Burkina_Faso', 'Bangladesh', 'Bulgaria', 'Bahrain', 'Bahamas', 'Bosnia_and_Herzegovina', 'Belarus', 'Belize', 'Bermuda', 'Bolivia', 'Brazil', 'Barbados', 'Brunei', 'Bhutan', 'Botswana', 'Central_African_Republic', 'Canada', 'Switzerland', 'Chile', 'China', "Cote_d'Ivoire", 'Cameroon', 'Democratic_Republic_of_Congo', 'Congo', 'Colombia', 'Cape_Verde', 'Costa_Rica', 'Cuba', 'Cyprus', 'Czech_Republic', 'Germany', 'Djibouti', 'Dominica', 'Denmark', 'Dominican_Republic', 'Algeria', 'Ecuador', 'Egypt', 'Eritrea', 'Spain', 'Estonia', 'Ethiopia', 'Finland', 'Fiji', 'France', 'Faeroe_Islands', 'Gabon', 'United_Kingdom', 'Georgia', 'Ghana', 'Guinea', 'Gambia', 'Greece', 'Greenland', 'Guatemala', 'Guam', 'Guyana', 'Hong_Kong', 'Honduras', 'Croatia', 'Haiti', 'Hungary', 'Indonesia', 'India', 'Ireland', 'Iran', 'Iraq', 'Iceland', 'Israel', 'Italy', 'Jamaica', 'Jordan', 'Japan', 'Kazakhstan', 'Kenya', 'Kyrgyz_Republic', 'Cambodia', 'South_Korea', 'Kuwait', 'Laos', 'Lebanon', 'Liberia', 'Libya', 'Liechtenstein', 'Sri_Lanka', 'Lithuania', 'Luxembourg', 'Latvia', 'Macao', 'Morocco', 'Monaco', 'Moldova', 'Madagascar', 'Mexico', 'Mali', 'Malta', 'Myanmar', 'Mongolia', 'Mozambique', 'Mauritania', 'Mauritius', 'Malaysia', 'Namibia', 'Niger', 'Nigeria', 'Nicaragua', 'Netherlands', 'Norway', 'Nepal', 'New_Zealand', 'Oman', 'Pakistan', 'Panama', 'Peru', 'Philippines', 'Papua_New_Guinea', 'Poland', 'Puerto_Rico', 'Portugal', 'Paraguay', 'Palestine', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saudi_Arabia', 'Sudan', 'Senegal', 'Singapore', 'Sierra_Leone', 'El_Salvador', 'San_Marino', 'Somalia', 'Serbia', 'Suriname', 'Slovak_Republic', 'Slovenia', 'Sweden', 'Eswatini', 'Seychelles', 'Syria', 'Chad', 'Togo', 'Thailand', 'Trinidad_and_Tobago', 'Tunisia', 'Turkey', 'Taiwan', 'Tanzania', 'Uganda', 'Ukraine', 'Uruguay', 'United_States', 'Uzbekistan', 'Venezuela', 'United_States_Virgin_Islands', 'Vietnam', 'South_Africa', 'Zambia', 'Zimbabwe']





#Finds the first non-zero confirmed case
def first_nonzero_case_index(cases):
	fnz = 0 
	for i in cases:
		if isnan(i) == False and i != 0:
			break
		fnz += 1	
	return fnz



def get_case(country_name):
	df = pd.read_csv('Data/OxCGRT_latest_cleaned.csv', index_col='Index')

	df_country = df.loc[df['CountryName'] == country_name]
	cases = df_country['ConfirmedCases'].to_numpy()[:-1]
	
	fnz = first_nonzero_case_index(cases)

	return int(cases[fnz+364])
	



def main():
	cases = {}
	for country in countries:
		cases[country] = get_case(country)	


	wc = WordCloud(background_color="white", random_state=1)
	wc.generate_from_frequencies(cases)

	plt.imshow(wc, interpolation="bilinear")
	plt.axis("off")
	plt.savefig('./cases_figs/word_cloud')
	plt.show()




main()
