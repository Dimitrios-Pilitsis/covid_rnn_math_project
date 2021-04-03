import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap


sns.set(style='white', context='poster', rc={'figure.figsize':(10,8)})

#countries=['Greece', 'Italy', 'United_Kingdom']

countries=['Aruba', 'Afghanistan', 'Angola', 'Albania', 'Andorra', 'United_Arab_Emirates', 'Argentina', 'Australia', 'Austria', 'Azerbaijan', 'Burundi', 'Belgium', 'Benin', 'Burkina_Faso', 'Bangladesh', 'Bulgaria', 'Bahrain', 'Bahamas', 'Bosnia_and_Herzegovina', 'Belarus', 'Belize', 'Bermuda', 'Bolivia', 'Brazil', 'Barbados', 'Brunei', 'Bhutan', 'Botswana', 'Central_African_Republic', 'Canada', 'Switzerland', 'Chile', 'China', "Cote_d'Ivoire", 'Cameroon', 'Democratic_Republic_of_Congo', 'Congo', 'Colombia', 'Cape_Verde', 'Costa_Rica', 'Cuba', 'Cyprus', 'Czech_Republic', 'Germany', 'Djibouti', 'Dominica', 'Denmark', 'Dominican_Republic', 'Algeria', 'Ecuador', 'Egypt', 'Eritrea', 'Spain', 'Estonia', 'Ethiopia', 'Finland', 'Fiji', 'France', 'Faeroe_Islands', 'Gabon', 'United_Kingdom', 'Georgia', 'Ghana', 'Guinea', 'Gambia', 'Greece', 'Greenland', 'Guatemala', 'Guam', 'Guyana', 'Hong_Kong', 'Honduras', 'Croatia', 'Haiti', 'Hungary', 'Indonesia', 'India', 'Ireland', 'Iran', 'Iraq', 'Iceland', 'Israel', 'Italy', 'Jamaica', 'Jordan', 'Japan', 'Kazakhstan', 'Kenya', 'Kyrgyz_Republic', 'Cambodia', 'South_Korea', 'Kuwait', 'Laos', 'Lebanon', 'Liberia', 'Libya', 'Liechtenstein', 'Sri_Lanka', 'Lithuania', 'Luxembourg', 'Latvia', 'Macao', 'Morocco', 'Monaco', 'Moldova', 'Madagascar', 'Mexico', 'Mali', 'Malta', 'Myanmar', 'Mongolia', 'Mozambique', 'Mauritania', 'Mauritius', 'Malaysia', 'Namibia', 'Niger', 'Nigeria', 'Nicaragua', 'Netherlands', 'Norway', 'Nepal', 'New_Zealand', 'Oman', 'Pakistan', 'Panama', 'Peru', 'Philippines', 'Papua_New_Guinea', 'Poland', 'Puerto_Rico', 'Portugal', 'Paraguay', 'Palestine', 'Qatar', 'Kosovo', 'Romania', 'Russia', 'Rwanda', 'Saudi_Arabia', 'Sudan', 'Senegal', 'Singapore', 'Sierra_Leone', 'El_Salvador', 'San_Marino', 'Somalia', 'Serbia', 'Suriname', 'Slovak_Republic', 'Slovenia', 'Sweden', 'Eswatini', 'Seychelles', 'Syria', 'Chad', 'Togo', 'Thailand', 'Timor-Leste', 'Trinidad_and_Tobago', 'Tunisia', 'Turkey', 'Taiwan', 'Tanzania', 'Uganda', 'Ukraine', 'Uruguay', 'United_States', 'Uzbekistan', 'Venezuela', 'United_States_Virgin_Islands', 'Vietnam', 'South_Africa', 'Zambia', 'Zimbabwe']


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


# Turn into numpy arrays and remove the last dimension which has axis of 1
backward_bias = np.squeeze(np.array(backward_bias))
backward_kernel = np.squeeze(np.array(backward_kernel))
dense_kernel = np.squeeze(np.array(dense_kernel))
forward_bias = np.squeeze(np.array(forward_bias))
forward_kernel = np.squeeze(np.array(forward_kernel))
"""
print(backward_bias.shape)
print(backward_kernel.shape)
print(dense_kernel.shape)
print(forward_bias.shape)
print(forward_kernel.shape)
"""

reducer_bb = umap.UMAP()
embedding_bb = reducer_bb.fit_transform(backward_bias)
print(embedding_bb.shape)

plt.scatter(embedding_bb[:, 0], embedding_bb[:, 1], c=np.arange(len(countries)), cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
#plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of backward_bias', fontsize=24)
plt.savefig('./umap_figs/backward_bias')
plt.show()




reducer_bk = umap.UMAP()
embedding_bk = reducer_bk.fit_transform(backward_kernel)
print(embedding_bk.shape)

plt.scatter(embedding_bk[:, 0], embedding_bk[:, 1], c=np.arange(len(countries)), cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
#plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of backward_kernel', fontsize=24)
plt.savefig('./umap_figs/backward_kernel')
plt.show()



reducer_dk = umap.UMAP()
embedding_dk = reducer_dk.fit_transform(dense_kernel)
print(embedding_dk.shape)

plt.scatter(embedding_dk[:, 0], embedding_dk[:, 1], c=np.arange(len(countries)), cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
#plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of dense_kernel', fontsize=24)
plt.savefig('./umap_figs/dense_kernel')
plt.show()



reducer_fb = umap.UMAP()
embedding_fb = reducer_fb.fit_transform(forward_bias)
print(embedding_fb.shape)

plt.scatter(embedding_fb[:, 0], embedding_fb[:, 1], c=np.arange(len(countries)), cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
#plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of forward_bias', fontsize=24)
plt.savefig('./umap_figs/forward_bias')
plt.show()



reducer_fk = umap.UMAP()
embedding_fk = reducer_fk.fit_transform(forward_kernel)
print(embedding_fk.shape)

plt.scatter(embedding_fk[:, 0], embedding_fk[:, 1], c=np.arange(len(countries)), cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
#plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of forward_kernel', fontsize=24)
plt.savefig('./umap_figs/forward_kernel')
plt.show()
