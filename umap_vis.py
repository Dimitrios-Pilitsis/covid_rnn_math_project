import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn import random_projection

# List of countries -----------------------------------------------------------------------

countries=['Aruba', 'Afghanistan', 'Angola', 'Albania', 'Andorra', 'United_Arab_Emirates', 'Argentina', 'Australia', 'Austria', 'Azerbaijan', 'Burundi', 'Belgium', 'Benin', 'Burkina_Faso', 'Bangladesh', 'Bulgaria', 'Bahrain', 'Bahamas', 'Bosnia_and_Herzegovina', 'Belarus', 'Belize', 'Bermuda', 'Bolivia', 'Brazil', 'Barbados', 'Brunei', 'Bhutan', 'Botswana', 'Central_African_Republic', 'Canada', 'Switzerland', 'Chile', 'China', "Cote_d'Ivoire", 'Cameroon', 'Democratic_Republic_of_Congo', 'Congo', 'Colombia', 'Cape_Verde', 'Costa_Rica', 'Cuba', 'Cyprus', 'Czech_Republic', 'Germany', 'Djibouti', 'Dominica', 'Denmark', 'Dominican_Republic', 'Algeria', 'Ecuador', 'Egypt', 'Eritrea', 'Spain', 'Estonia', 'Ethiopia', 'Finland', 'Fiji', 'France', 'Faeroe_Islands', 'Gabon', 'United_Kingdom', 'Georgia', 'Ghana', 'Guinea', 'Gambia', 'Greece', 'Greenland', 'Guatemala', 'Guam', 'Guyana', 'Hong_Kong', 'Honduras', 'Croatia', 'Haiti', 'Hungary', 'Indonesia', 'India', 'Ireland', 'Iran', 'Iraq', 'Iceland', 'Israel', 'Italy', 'Jamaica', 'Jordan', 'Japan', 'Kazakhstan', 'Kenya', 'Kyrgyz_Republic', 'Cambodia', 'South_Korea', 'Kuwait', 'Laos', 'Lebanon', 'Liberia', 'Libya', 'Liechtenstein', 'Sri_Lanka', 'Lithuania', 'Luxembourg', 'Latvia', 'Macao', 'Morocco', 'Monaco', 'Moldova', 'Madagascar', 'Mexico', 'Mali', 'Malta', 'Myanmar', 'Mongolia', 'Mozambique', 'Mauritania', 'Mauritius', 'Malaysia', 'Namibia', 'Niger', 'Nigeria', 'Nicaragua', 'Netherlands', 'Norway', 'Nepal', 'New_Zealand', 'Oman', 'Pakistan', 'Panama', 'Peru', 'Philippines', 'Papua_New_Guinea', 'Poland', 'Puerto_Rico', 'Portugal', 'Paraguay', 'Palestine', 'Qatar', 'Kosovo', 'Romania', 'Russia', 'Rwanda', 'Saudi_Arabia', 'Sudan', 'Senegal', 'Singapore', 'Sierra_Leone', 'El_Salvador', 'San_Marino', 'Somalia', 'Serbia', 'Suriname', 'Slovak_Republic', 'Slovenia', 'Sweden', 'Eswatini', 'Seychelles', 'Syria', 'Chad', 'Togo', 'Thailand', 'Timor-Leste', 'Trinidad_and_Tobago', 'Tunisia', 'Turkey', 'Taiwan', 'Tanzania', 'Uganda', 'Ukraine', 'Uruguay', 'United_States', 'Uzbekistan', 'Venezuela', 'United_States_Virgin_Islands', 'Vietnam', 'South_Africa', 'Zambia', 'Zimbabwe']


countries_eu=['Austria','Italy','Belgium','Latvia','Bulgaria','Lithuania','Croatia','Luxembourg','Cyprus','Malta','Czechia','Netherlands','Denmark','Poland','Estonia','Portugal','Finland','Romania','France','Slovakia','Germany','Slovenia','Greece','Spain','Hungary','Sweden','Ireland']

countries_main = ['United_Arab_Emirates', 'Australia', 'Brazil', 'Canada', 'Chile', 'Germany', 'Greece', 'Spain', 'France', 'Israel', 'Italy', 'Japan', 'New_Zealand', 'South_Korea', 'United_States', 'South_Africa'] 


#Colors -----------------------------------------------------------------------

colors_main = ['black', 'gray', 'brown', 'r', 'lightsalmon', 'saddlebrown', 'orange', 'olive', 'yellow', 'g', 'lime', 'turquoise', 'cyan', 'navy', 'b', 'purple', 'magenta', 'm', 'pink']
	



"""
# Generic load data ----------------------------------------------------------------------------------------------------------
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
	backward_recurrent_kernel.append(np.loadtxt('./weights/' + country + '/backward_recurrent_kernel.txt').reshape(1,-1))
	dense_kernel.append(np.loadtxt('./weights/' + country + '/dense_kernel.txt').reshape((128,1)))
	forward_bias.append(np.loadtxt('./weights/' + country + '/forward_bias.txt').reshape((256,1)))
	forward_kernel.append(np.loadtxt('./weights/' + country + '/forward_kernel.txt').reshape((256,1)))
	forward_recurrent_kernel.append(np.loadtxt('./weights/' + country + '/forward_recurrent_kernel.txt').reshape(1,-1))


# Turn into numpy arrays and remove the last dimension which has axis of 1
backward_bias = np.squeeze(np.array(backward_bias))
backward_kernel = np.squeeze(np.array(backward_kernel))
dense_kernel = np.squeeze(np.array(dense_kernel))
forward_bias = np.squeeze(np.array(forward_bias))
forward_kernel = np.squeeze(np.array(forward_kernel))
forward_recurrent_kernel = np.squeeze(np.array(forward_recurrent_kernel))
backward_recurrent_kernel = np.squeeze(np.array(backward_recurrent_kernel))
"""
# Main countries load data -------------------------------------------------------------------------------
brk_main = []
frk_main = []
for country in countries_main:
	brk_main.append(np.loadtxt('./weights/' + country + '/backward_recurrent_kernel.txt').reshape(1,-1))
	frk_main.append(np.loadtxt('./weights/' + country + '/forward_recurrent_kernel.txt').reshape(1,-1))


brk_main = np.squeeze(np.array(brk_main))
frk_main = np.squeeze(np.array(frk_main))




# UMAP visualizations -------------------------------------------------------------------------------------

def umap_vis_basic(dataset, countries, name):
	reducer = umap.UMAP()
	embedding = reducer.fit_transform(dataset)

	plt.scatter(embedding[:, 0], embedding[:, 1], c=np.arange(len(countries)), cmap='Spectral', s=5)
	plt.gca().set_aspect('equal', 'datalim')
	#plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
	plt.title('UMAP projection ' + name + ' all countries', fontsize=12)
	plt.savefig('./umap_figs/' + name + '_all_countries')
	plt.show()


def vis_main_countries(dataset, countries, colors, name):
	reducer = umap.UMAP()
	embedding = reducer.fit_transform(dataset)

	fig, ax = plt.subplots(figsize=(10.0, 5.0))
	for i,emb in enumerate(embedding):
		ax.scatter(emb[0], emb[1], c=colors[i], label=countries[i]) 


	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.subplots_adjust(left=0.1, bottom=0.1, right=0.6, top=0.8)
	plt.title('UMAP projection of ' + name + ' main', fontsize=12)
	plt.savefig('./umap_figs/' + name + '_main')
	plt.show()


#Johnson-Lindenstrauss---------------------------------------------------------------------------

def jl_transform_vis(dataset, countries, colors, name):
	transformer = random_projection.GaussianRandomProjection()
	#transformer_1 = random_projection.SparseRandomProjection()
	dataset_transformed = transformer.fit_transform(dataset)

	vis_main_countries(dataset_transformed, countries_main, colors_main, name)

"""
jl_transform_vis(brk_main, countries, colors_main, 'backward_recurrent_kernel under JL transformation')
jl_transform_vis(frk_main, countries, colors_main, 'forward_recurrent_kernel under JL transformation')
"""




# Visualization calls ----------------------------------------------------------------
vis_main_countries(brk_main, countries_main, colors_main, 'backward_recurrent_kernel')
vis_main_countries(frk_main, countries_main, colors_main, 'forward_recurrent_kernel')

"""
umap_vis_basic(backward_bias, 'backward_bias')
umap_vis_basic(backward_kernel, 'backward_kernel')
umap_vis_basic(dense_kernel, 'dense_kernel')
umap_vis_basic(forward_bias, 'forward_bias')
umap_vis_basic(forward_kernel, 'forward_kernel')
"""



"""
umap_vis_basic(forward_recurrent_kernel, 'forward_recurrent_kernel')
umap_vis_basic(backward_recurrent_kernel, 'backward_recurrent_kernel')

"""
