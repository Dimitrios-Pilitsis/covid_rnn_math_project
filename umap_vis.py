import os
import numpy as np
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
"""
#sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})

countries=['Greece', 'Italy', 'United_Kingdom']

#countries=['Aruba', 'Afghanistan', 'Angola', 'Albania', 'Andorra', 'United_Arab_Emirates', 'Argentina', 'Australia', 'Austria', 'Azerbaijan', 'Burundi', 'Belgium', 'Benin', 'Burkina_Faso', 'Bangladesh', 'Bulgaria', 'Bahrain', 'Bahamas', 'Bosnia_and_Herzegovina', 'Belarus', 'Belize', 'Bermuda', 'Bolivia', 'Brazil', 'Barbados', 'Brunei', 'Bhutan', 'Botswana', 'Central_African_Republic', 'Canada', 'Switzerland', 'Chile', 'China', "Cote_d'Ivoire", 'Cameroon', 'Democratic_Republic_of_Congo', 'Congo', 'Colombia', 'Comoros', 'Cape_Verde', 'Costa_Rica', 'Cuba', 'Cyprus', 'Czech_Republic', 'Germany', 'Djibouti', 'Dominica', 'Denmark', 'Dominican_Republic', 'Algeria', 'Ecuador', 'Egypt', 'Eritrea', 'Spain', 'Estonia', 'Ethiopia', 'Finland', 'Fiji', 'France', 'Faeroe_Islands', 'Gabon', 'United_Kingdom', 'Georgia', 'Ghana', 'Guinea', 'Gambia', 'Greece', 'Greenland', 'Guatemala', 'Guam', 'Guyana', 'Hong_Kong', 'Honduras', 'Croatia', 'Haiti', 'Hungary', 'Indonesia', 'India', 'Ireland', 'Iran', 'Iraq', 'Iceland', 'Israel', 'Italy', 'Jamaica', 'Jordan', 'Japan', 'Kazakhstan', 'Kenya', 'Kyrgyz_Republic', 'Cambodia', 'Kiribati', 'South_Korea', 'Kuwait', 'Laos', 'Lebanon', 'Liberia', 'Libya', 'Liechtenstein', 'Sri_Lanka', 'Lesotho', 'Lithuania', 'Luxembourg', 'Latvia', 'Macao', 'Morocco', 'Monaco', 'Moldova', 'Madagascar', 'Mexico', 'Mali', 'Malta', 'Myanmar', 'Mongolia', 'Mozambique', 'Mauritania', 'Mauritius', 'Malawi', 'Malaysia', 'Namibia', 'Niger', 'Nigeria', 'Nicaragua', 'Netherlands', 'Norway', 'Nepal', 'New_Zealand', 'Oman', 'Pakistan', 'Panama', 'Peru', 'Philippines', 'Papua_New_Guinea', 'Poland', 'Puerto_Rico', 'Portugal', 'Paraguay', 'Palestine', 'Qatar', 'Kosovo', 'Romania', 'Russia', 'Rwanda', 'Saudi_Arabia', 'Sudan', 'Senegal', 'Singapore', 'Solomon_Islands', 'Sierra_Leone', 'El_Salvador', 'San_Marino', 'Somalia', 'Serbia', 'South_Sudan', 'Suriname', 'Slovak_Republic', 'Slovenia', 'Sweden', 'Eswatini', 'Seychelles', 'Syria', 'Chad', 'Togo', 'Thailand', 'Tajikistan', 'Turkmenistan', 'Timor-Leste', 'Tonga', 'Trinidad_and_Tobago', 'Tunisia', 'Turkey', 'Taiwan', 'Tanzania', 'Uganda', 'Ukraine', 'Uruguay', 'United_States', 'Uzbekistan', 'Venezuela', 'United_States_Virgin_Islands', 'Vietnam', 'Vanuatu', 'Yemen', 'South_Africa', 'Zambia', 'Zimbabwe']



backward_bias = []
backward_kernel = []
backward_recurrent_kernel = []
dense_kernel = []
forward_bias = []
forward_kernel = []
forward_recurrent_kernel = []


for country in countries:
	backward_bias.append(np.loadtxt('./weights/' + country + '/backward_bias.txt').reshape((256,1)))
	backward_kernel.append(np.loadtxt('./weights/' + country + '/backward_kernel.txt').reshape((256,1)))
	backward_recurrent_kernel.append(np.loadtxt('./weights/' + country + '/backward_recurrent_kernel.txt'))
	dense_kernel.append(np.loadtxt('./weights/' + country + '/dense_kernel.txt').reshape((128,1)))
	forward_bias.append(np.loadtxt('./weights/' + country + '/forward_bias.txt').reshape((256,1)))
	forward_kernel.append(np.loadtxt('./weights/' + country + '/forward_kernel.txt').reshape((256,1)))
	forward_recurrent_kernel.append(np.loadtxt('./weights/' + country + '/forward_recurrent_kernel.txt'))



print(len(backward_bias))
print(len(backward_kernel))
print(len(backward_recurrent_kernel))
print(len(dense_kernel))
print(len(forward_bias))
print(len(forward_kernel))
print(len(forward_recurrent_kernel))



"""
reducer = umap.UMAP()
embedding = reducer.fit_transform(scaled_penguin_data)
print(embedding.shape)


plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[sns.color_palette()[x] for x in penguins.species_short.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Penguin dataset', fontsize=24)
plt.show()

plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset', fontsize=24);
plt.show()
"""
