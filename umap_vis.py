import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn import random_projection

countries=['Aruba', 'Afghanistan', 'Angola', 'Albania', 'Andorra', 'United_Arab_Emirates', 'Argentina', 'Australia', 'Austria', 'Azerbaijan', 'Burundi', 'Belgium', 'Benin', 'Burkina_Faso', 'Bangladesh', 'Bulgaria', 'Bahrain', 'Bahamas', 'Bosnia_and_Herzegovina', 'Belarus', 'Belize', 'Bermuda', 'Bolivia', 'Brazil', 'Barbados', 'Brunei', 'Bhutan', 'Botswana', 'Central_African_Republic', 'Canada', 'Switzerland', 'Chile', 'China', "Cote_d'Ivoire", 'Cameroon', 'Democratic_Republic_of_Congo', 'Congo', 'Colombia', 'Cape_Verde', 'Costa_Rica', 'Cuba', 'Cyprus', 'Czech_Republic', 'Germany', 'Djibouti', 'Dominica', 'Denmark', 'Dominican_Republic', 'Algeria', 'Ecuador', 'Egypt', 'Eritrea', 'Spain', 'Estonia', 'Ethiopia', 'Finland', 'Fiji', 'France', 'Faeroe_Islands', 'Gabon', 'United_Kingdom', 'Georgia', 'Ghana', 'Guinea', 'Gambia', 'Greece', 'Greenland', 'Guatemala', 'Guam', 'Guyana', 'Hong_Kong', 'Honduras', 'Croatia', 'Haiti', 'Hungary', 'Indonesia', 'India', 'Ireland', 'Iran', 'Iraq', 'Iceland', 'Israel', 'Italy', 'Jamaica', 'Jordan', 'Japan', 'Kazakhstan', 'Kenya', 'Kyrgyz_Republic', 'Cambodia', 'South_Korea', 'Kuwait', 'Laos', 'Lebanon', 'Liberia', 'Libya', 'Liechtenstein', 'Sri_Lanka', 'Lithuania', 'Luxembourg', 'Latvia', 'Macao', 'Morocco', 'Monaco', 'Moldova', 'Madagascar', 'Mexico', 'Mali', 'Malta', 'Myanmar', 'Mongolia', 'Mozambique', 'Mauritania', 'Mauritius', 'Malaysia', 'Namibia', 'Niger', 'Nigeria', 'Nicaragua', 'Netherlands', 'Norway', 'Nepal', 'New_Zealand', 'Oman', 'Pakistan', 'Panama', 'Peru', 'Philippines', 'Papua_New_Guinea', 'Poland', 'Puerto_Rico', 'Portugal', 'Paraguay', 'Palestine', 'Qatar', 'Kosovo', 'Romania', 'Russia', 'Rwanda', 'Saudi_Arabia', 'Sudan', 'Senegal', 'Singapore', 'Sierra_Leone', 'El_Salvador', 'San_Marino', 'Somalia', 'Serbia', 'Suriname', 'Slovak_Republic', 'Slovenia', 'Sweden', 'Eswatini', 'Seychelles', 'Syria', 'Chad', 'Togo', 'Thailand', 'Timor-Leste', 'Trinidad_and_Tobago', 'Tunisia', 'Turkey', 'Taiwan', 'Tanzania', 'Uganda', 'Ukraine', 'Uruguay', 'United_States', 'Uzbekistan', 'Venezuela', 'United_States_Virgin_Islands', 'Vietnam', 'South_Africa', 'Zambia', 'Zimbabwe']




countries_eu=['Austria','Italy','Belgium','Latvia','Bulgaria','Lithuania','Croatia','Luxembourg','Cyprus','Malta','Czechia','Netherlands','Denmark','Poland','Estonia','Portugal','Finland','Romania','France','Slovakia','Germany','Slovenia','Greece','Spain','Hungary','Sweden','Ireland']

"""
# Load data into arrays ------------------------------------------------------------------
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
# UMAP visualizations -------------------------------------------------------------------------------------

def umap_vis_basic(dataset, name):
	reducer = umap.UMAP()
	embedding = reducer.fit_transform(dataset)
	print(type(embedding))
	print(embedding.shape)

	plt.scatter(embedding[:, 0], embedding[:, 1], c=np.arange(len(countries)), cmap='Spectral', s=5)
	plt.gca().set_aspect('equal', 'datalim')
	#plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
	plt.title('UMAP projection of ' + name, fontsize=24)
	plt.savefig('./umap_figs/' + name)
	plt.show()


def vis_interesting_countries():

	countries_interesting = ['United_Arab_Emirates', 'Australia', 'Brazil', 'Canada', 'Chile', 'Germany', 'Greece', 'Spain', 'France', 'Israel', 'Italy', 'Japan', 'New_Zealand', 'South_Korea', 'United_States', 'South_Africa'] 

	colors = ['black', 'gray', 'brown', 'r', 'lightsalmon', 'saddlebrown', 'orange', 'olive', 'yellow', 'g', 'lime', 'turquoise', 'cyan', 'deepskyblue', 'navy', 'b', 'purple', 'magenta', 'pink']

	brk_interesting = []
	for country in countries_interesting:
		brk_interesting.append(np.loadtxt('./weights/' + country + '/backward_recurrent_kernel.txt').reshape(1,-1))


	brk_interesting = np.squeeze(np.array(brk_interesting))

	reducer = umap.UMAP()
	embedding = reducer.fit_transform(brk_interesting)

	fig, ax = plt.subplots()
	for i,emb in enumerate(embedding):
		ax.scatter(emb[0], emb[1], c=colors[i], label=countries_interesting[i]) 


	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.subplots_adjust(left=0.1, bottom=0.1, right=0.6, top=0.8)
	plt.title('UMAP projection of backward_recurrent_kernel subsection', fontsize=24)
	plt.show()



vis_interesting_countries()


"""
umap_vis_basic(backward_bias, 'backward_bias')
umap_vis_basic(backward_kernel, 'backward_kernel')
umap_vis_basic(dense_kernel, 'dense_kernel')
umap_vis_basic(forward_bias, 'forward_bias')
umap_vis_basic(forward_kernel, 'forward_kernel')
umap_vis_basic(forward_recurrent_kernel, 'forward_recurrent_kernel')
umap_vis_basic(backward_recurrent_kernel, 'backward_recurrent_kernel')
"""


"""
#Johnson-Lindenstrauss---------------------------------------------------------------------------
transformer_1 = random_projection.GaussianRandomProjection()
#transformer_1 = random_projection.SparseRandomProjection()
frk_transformed = transformer_1.fit_transform(forward_recurrent_kernel)

transformer_2 = random_projection.GaussianRandomProjection()
#transformer_2 = random_projection.SparseRandomProjection()
brk_transformed = transformer_2.fit_transform(backward_recurrent_kernel)


umap_vis_basic(frk_transformed, 'forward_recurrent_kernel under Johnson–Lindenstrauss transformation')
umap_vis_basic(brk_transformed, 'backward_recurrent_kernel under Johnson–Lindenstrauss transformation')
"""
