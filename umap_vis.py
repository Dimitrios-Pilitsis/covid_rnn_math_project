import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pt
import umap
from sklearn import random_projection
import pycountry_convert as pc


# List of countries -----------------------------------------------------------------------

countries_world=['Aruba', 'Afghanistan', 'Angola', 'Albania', 'Andorra', 'United_Arab_Emirates', 'Argentina', 'Australia', 'Austria', 'Azerbaijan', 'Burundi', 'Belgium', 'Benin', 'Burkina_Faso', 'Bangladesh', 'Bulgaria', 'Bahrain', 'Bahamas', 'Bosnia_and_Herzegovina', 'Belarus', 'Belize', 'Bermuda', 'Bolivia', 'Brazil', 'Barbados', 'Brunei', 'Bhutan', 'Botswana', 'Central_African_Republic', 'Canada', 'Switzerland', 'Chile', 'China', "Cote_d'Ivoire", 'Cameroon', 'Democratic_Republic_of_Congo', 'Congo', 'Colombia', 'Cape_Verde', 'Costa_Rica', 'Cuba', 'Cyprus', 'Czech_Republic', 'Germany', 'Djibouti', 'Dominica', 'Denmark', 'Dominican_Republic', 'Algeria', 'Ecuador', 'Egypt', 'Eritrea', 'Spain', 'Estonia', 'Ethiopia', 'Finland', 'Fiji', 'France', 'Faeroe_Islands', 'Gabon', 'United_Kingdom', 'Georgia', 'Ghana', 'Guinea', 'Gambia', 'Greece', 'Greenland', 'Guatemala', 'Guam', 'Guyana', 'Hong_Kong', 'Honduras', 'Croatia', 'Haiti', 'Hungary', 'Indonesia', 'India', 'Ireland', 'Iran', 'Iraq', 'Iceland', 'Israel', 'Italy', 'Jamaica', 'Jordan', 'Japan', 'Kazakhstan', 'Kenya', 'Kyrgyz_Republic', 'Cambodia', 'South_Korea', 'Kuwait', 'Laos', 'Lebanon', 'Liberia', 'Libya', 'Liechtenstein', 'Sri_Lanka', 'Lithuania', 'Luxembourg', 'Latvia', 'Macao', 'Morocco', 'Monaco', 'Moldova', 'Madagascar', 'Mexico', 'Mali', 'Malta', 'Myanmar', 'Mongolia', 'Mozambique', 'Mauritania', 'Mauritius', 'Malaysia', 'Namibia', 'Niger', 'Nigeria', 'Nicaragua', 'Netherlands', 'Norway', 'Nepal', 'New_Zealand', 'Oman', 'Pakistan', 'Panama', 'Peru', 'Philippines', 'Papua_New_Guinea', 'Poland', 'Puerto_Rico', 'Portugal', 'Paraguay', 'Palestine', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saudi_Arabia', 'Sudan', 'Senegal', 'Singapore', 'Sierra_Leone', 'El_Salvador', 'San_Marino', 'Somalia', 'Serbia', 'Suriname', 'Slovak_Republic', 'Slovenia', 'Sweden', 'Eswatini', 'Seychelles', 'Syria', 'Chad', 'Togo', 'Thailand', 'Trinidad_and_Tobago', 'Tunisia', 'Turkey', 'Taiwan', 'Tanzania', 'Uganda', 'Ukraine', 'Uruguay', 'United_States', 'Uzbekistan', 'Venezuela', 'United_States_Virgin_Islands', 'Vietnam', 'South_Africa', 'Zambia', 'Zimbabwe']

def get_continent_each_country(countries):
	countries_without_underscore = [country.replace("_", " ") for country in countries]
	countries_namechange = {"Cote d'Ivoire" : "Ivory Coast", "Democratic Republic of Congo": "Democratic Republic of the Congo", "Faeroe Islands" : "Faroe Islands"}

	for country in countries_namechange:
		index =	countries_without_underscore.index(country)
		countries_without_underscore[index] = countries_namechange[country] 


	continents_of_countries = []

	for country in countries_without_underscore:
		country_code = pc.country_name_to_country_alpha2(country, cn_name_format="default")
		continent_name = pc.country_alpha2_to_continent_code(country_code)
		continents_of_countries.append(continent_name)

	return continents_of_countries


continents_world = get_continent_each_country(countries_world)


countries_eu=['Austria','Italy','Belgium','Latvia','Bulgaria','Lithuania','Croatia','Luxembourg','Cyprus','Malta','Czechia','Netherlands','Denmark','Poland','Estonia','Portugal','Finland','Romania','France','Slovakia','Germany','Slovenia','Greece','Spain','Hungary','Sweden','Ireland']

countries_main = ['United_Arab_Emirates', 'Australia', 'Brazil', 'Canada', 'Chile', 'Germany', 'Greece', 'Spain', 'France', 'Israel', 'Italy', 'Japan', 'New_Zealand', 'South_Korea', 'United_States', 'South_Africa'] 


#Colors -----------------------------------------------------------------------

colors_main = ['black', 'gray', 'brown', 'r', 'lightsalmon', 'saddlebrown', 'orange', 'olive', 'yellow', 'g', 'lime', 'turquoise', 'cyan', 'navy', 'b', 'purple', 'magenta', 'm', 'pink']
	

# Convert continents from codes to names
continents_convert = {"AF" : "Africa", "EU" : "European Union", "NA" : "North America" , "SA" : "South America", "AS" : "Asia", "OC" : "Oceania"}
	
for i in range(len(continents_world)):
	continents_world[i] = continents_convert[continents_world[i]]	

colors_world_map = {"Africa" : 'black', "European Union" : 'b', "North America" : 'yellow', "South America" : 'g', "Asia" : 'r', "Oceania" : 'm'} 
colors_world = []


for c in continents_world:
	colors_world.append(colors_world_map[c])


# World data----------------------------------------------------------------------------------------------------------

def get_world_data():
	backward_bias = []
	backward_kernel = []
	backward_recurrent_kernel = []
	dense_kernel = []
	forward_bias = []
	forward_kernel = []
	forward_recurrent_kernel = []

	for country in countries_world:
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

	return backward_bias, backward_kernel, dense_kernel, forward_bias, forward_kernel, backward_recurrent_kernel, forward_recurrent_kernel

# Main countries data -------------------------------------------------------------------------------

def get_main_data():
	bb_main = []
	bk_main = []
	dk_main = []
	fb_main = []
	fk_main = []
	brk_main = []
	frk_main = []

	for country in countries_main:
		bb_main.append(np.loadtxt('./weights/' + country + '/backward_bias.txt').reshape((256,1)))
		bk_main.append(np.loadtxt('./weights/' + country + '/backward_kernel.txt').reshape((256,1)))
		dk_main.append(np.loadtxt('./weights/' + country + '/dense_kernel.txt').reshape((128,1)))
		fb_main.append(np.loadtxt('./weights/' + country + '/forward_bias.txt').reshape((256,1)))
		fk_main.append(np.loadtxt('./weights/' + country + '/forward_kernel.txt').reshape((256,1)))
		brk_main.append(np.loadtxt('./weights/' + country + '/backward_recurrent_kernel.txt').reshape(1,-1))
		frk_main.append(np.loadtxt('./weights/' + country + '/forward_recurrent_kernel.txt').reshape(1,-1))

	bb_main = np.squeeze(np.array(bb_main))
	bk_main = np.squeeze(np.array(bk_main))
	dk_main = np.squeeze(np.array(dk_main))
	fb_main = np.squeeze(np.array(fb_main))
	fk_main = np.squeeze(np.array(fk_main))
	brk_main = np.squeeze(np.array(brk_main))
	frk_main = np.squeeze(np.array(frk_main))

	return bb_main, bk_main, dk_main, fb_main, fk_main, brk_main, frk_main


# UMAP visualizations -------------------------------------------------------------------------------------

def umap_vis_basic(dataset, continents, colors, name):
	reducer = umap.UMAP()
	embedding = reducer.fit_transform(dataset)

	fig, ax = plt.subplots(figsize=(10.0, 5.0))
	for i,emb in enumerate(embedding):
		ax.scatter(emb[0], emb[1], c=colors[i]) 

	colors_legend = ['black', 'b', 'yellow', 'g', 'r','m'] 
	continents_legend = ["Africa", "European Union", "North America", "South America", "Asia", "Oceania"] 
		
	legend_content = [pt.Patch(color=colors_legend[i], label=continents_legend[i]) for i in range(len(continents_legend))]
	
	plt.subplots_adjust(left=0.1, bottom=0.1, right=0.6, top=0.8)
	plt.title('UMAP projection of ' + name + ' all countries', fontsize=12)
	plt.legend(handles=legend_content, loc='center left', bbox_to_anchor=(1, 0.5))
	plt.savefig('./umap_figs/' + name + '_all_countries')
	#plt.show()



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
	#plt.show()


#Johnson-Lindenstrauss---------------------------------------------------------------------------

def jl_transform_gaussian(dataset):
	transformer = random_projection.GaussianRandomProjection()
	return transformer.fit_transform(dataset)

def jl_transform_sparse(dataset):
	transformer = random_projection.SparseRandomProjection()
	return transformer.fit_transform(dataset)


# Visualization calls ----------------------------------------------------------------


backward_bias, backward_kernel, dense_kernel, forward_bias, forward_kernel, backward_recurrent_kernel, forward_recurrent_kernel = get_world_data()
bb_main, bk_main, dk_main, fb_main, fk_main, brk_main, frk_main = get_main_data()


def world_vis(backward_bias, backward_kernel, dense_kernel, forward_bias, forward_kernel, backward_recurrent_kernel, forward_recurrent_kernel):
	umap_vis_basic(backward_bias, continents_world, colors_world,'backward_bias')
	umap_vis_basic(backward_kernel, continents_world, colors_world,'backward_kernel')
	umap_vis_basic(dense_kernel, continents_world, colors_world,'dense_kernel')
	umap_vis_basic(forward_bias, continents_world, colors_world,'forward_bias')
	umap_vis_basic(forward_kernel, continents_world, colors_world, 'forward_kernel')
	umap_vis_basic(forward_recurrent_kernel, continents_world, colors_world, 'forward_recurrent_kernel')
	umap_vis_basic(backward_recurrent_kernel, countries_world, colors_world, 'backward_recurrent_kernel')



def main_vis(bb_main, bk_main, dk_main, fb_main, fk_main, brk_main, frk_main):
	vis_main_countries(brk_main, countries_main, colors_main, 'backward_bias')
	vis_main_countries(brk_main, countries_main, colors_main, 'backward_kernel')
	vis_main_countries(brk_main, countries_main, colors_main, 'dense_kernel')
	vis_main_countries(brk_main, countries_main, colors_main, 'forward_bias')
	vis_main_countries(brk_main, countries_main, colors_main, 'forward_kernel')
	vis_main_countries(brk_main, countries_main, colors_main, 'backward_recurrent_kernel')
	vis_main_countries(frk_main, countries_main, colors_main, 'forward_recurrent_kernel')




def jl_vis_world(backward_recurrent_kernel, forward_recurrent_kernel, countries_world, colors_world):
	brk_l = jl_transform_gaussian(backward_recurrent_kernel)
	frk_l = jl_transform_gaussian(forward_recurrent_kernel)
	umap_vis_basic(brk_l, countries_world, colors_world, 'backward_recurrent_kernel_JLG')
	umap_vis_basic(frk_l, continents_world, colors_world, 'forward_recurrent_kernel_JLG')


	brk_s = jl_transform_sparse(backward_recurrent_kernel)
	frk_s = jl_transform_sparse(forward_recurrent_kernel)	
	umap_vis_basic(brk_s, countries_world, colors_world, 'backward_recurrent_kernel_JLS')
	umap_vis_basic(frk_s, continents_world, colors_world, 'forward_recurrent_kernel_JLS')



def jl_vis_main(brk_main, frk_main, countries_main, colors_main):
	brk_l = jl_transform_gaussian(brk_main)
	frk_l = jl_transform_gaussian(frk_main)
	vis_main_countries(brk_l, countries_main, colors_main, 'backward_recurrent_kernel_JLG')
	vis_main_countries(frk_l, countries_main, colors_main, 'forward_recurrent_kernel_JLG')

	brk_s = jl_transform_sparse(brk_main)
	frk_s = jl_transform_sparse(frk_main)	
	vis_main_countries(brk_s, countries_main, colors_main, 'backward_recurrent_kernel_JLS')
	vis_main_countries(frk_s, countries_main, colors_main, 'forward_recurrent_kernel_JLS')





world_vis(backward_bias, backward_kernel, dense_kernel, forward_bias, forward_kernel, backward_recurrent_kernel, forward_recurrent_kernel)
main_vis(bb_main, bk_main, dk_main, fb_main, fk_main, brk_main, frk_main)


jl_vis_world(backward_recurrent_kernel, forward_recurrent_kernel, countries_world, colors_world)
jl_vis_main(brk_main, frk_main, countries_main, colors_main)
