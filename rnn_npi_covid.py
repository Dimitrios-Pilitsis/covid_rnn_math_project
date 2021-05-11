import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from math import isnan
from sklearn.preprocessing import MinMaxScaler


#--------------------------------------------------------------
#Data wrangling

#Creates the timeseries so that it can be handled by TF
def setup_ts(df, country_name):
	df_country = df.loc[df['CountryName'] == country_name]
	df_country.loc[:,'Date'] = pd.to_datetime(df_country.loc[:, 'Date'], format='%Y%m%d')
	#df_greece['Date'] = pd.to_datetime(df_greece['Date'], format='%Y%m%d')
	ts_country = df_country.set_index('Date')
	ts_country = ts_country.drop(columns=['CountryName', 'CountryCode'])
	return ts_country



#Finds the first non-zero confirmed case
def first_nonzero_case_index(timeseries):
	confirmed_cases = timeseries['ConfirmedCases'].to_numpy()
	fnz = 0 
	for i in confirmed_cases:
		if isnan(i) == False and i != 0:
			break
		fnz += 1	
	return fnz



def setup_dataset(country_name):
	df = pd.read_csv('Data/OxCGRT_latest_cleaned.csv', index_col='Index')

	#Convert the csv into an appropriate timeseries
	ts = setup_ts(df, country_name)


	#Update timeseries so that the first value is the first confirmed case
	fnz = first_nonzero_case_index(ts)

	num_of_days_of_data = 365
	
	ts = ts[fnz:fnz+num_of_days_of_data] 


	#2D numpy array (doesn't include date column within the array)
	dataset_full = ts.to_numpy() 

	#Rescale the ConfirmedCases 
	#Create 2 different scalers as it makes it easier for later when plotting different graphs
	cases_scaler = MinMaxScaler()

	#Inner reshape is to satisfy sklearn, outer reshape is to keep the shape of array the same as originally
	dataset_full[:,-2] = cases_scaler.fit_transform(dataset_full[:,-2].reshape(-1,1))[:,0]

	#Timeseries is useful for any initial visualizations, dataset is for the TF model


	time = np.arange(len(dataset_full), dtype="float32") #Time is represented as day x since first covid case
	cases = dataset_full[:,-2]
	return time, cases, cases_scaler 



def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1:]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset




#Model functions-----------------------------------------------------------------------------------------

def learning_rate_optimizer(dataset):
	#print(dataset)
	#for x, y in dataset:
	#   print(x, y)

	model = tf.keras.models.Sequential([
		 tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
			      input_shape=[None]),
		 tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
		 tf.keras.layers.Dense(1),
	])


	#-----------------------------------------------------------------------
	#Learning rate Scheduker
	lr_schedule = tf.keras.callbacks.LearningRateScheduler(
	    lambda epoch: 1e-8 * 10**(epoch / 20))
	optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
	model.compile(loss=tf.keras.losses.Huber(),
		      optimizer=optimizer,
		      metrics=["mae"])
	history = model.fit(dataset, epochs=150, callbacks=[lr_schedule])

	#find minimum of loss
	optimal_lr = history.history["lr"][np.argmin(history.history["loss"])]

	#plt.semilogx(history.history["lr"], history.history["loss"])
	#plt.show()
	return optimal_lr
	

def run_model(dataset, learning_rate_optimal, epochs):
	model = tf.keras.models.Sequential([
		 tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
			      input_shape=[None]),
		 tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
		 tf.keras.layers.Dense(1),
	])


	model.summary()

	model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(lr=learning_rate_optimal, momentum=0.9),metrics=["mae"])

	history = model.fit(dataset, epochs=epochs)

	return history, model




def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast



#Visualizations functions ----------------------------------------------------------------

def basic_visualizations(time_valid, x_valid, results, history, epochs, cases_scaler):
	#For visualizations, we need to unscale the ConfirmedCases
	x_valid_unscaled = cases_scaler.inverse_transform(x_valid.reshape(-1, 1))[:,0]
	results_unscaled = cases_scaler.inverse_transform(results.reshape(-1, 1))[:,0]

	#Plot x_valid and predicted results on same graph to see how similar they are
	plt.figure(figsize=(10, 6))
	plt.plot(time_valid, x_valid_unscaled, 'r-')
	plt.plot(time_valid, results_unscaled, 'b-')
	plt.title("Validation and Results for confirmed cases")
	plt.xlabel("Time")
	plt.ylabel("Confirmed Cases")
	plt.legend(["Validation", "Predicted"])
	plt.show()

	#-----------------------------------------------------------
	# Retrieve a list of list results on training and test data
	# sets for each training epoch
	#-----------------------------------------------------------
	mae=history.history['mae']
	loss=history.history['loss']

	epochs=range(len(loss)) # Get number of epochs

	#------------------------------------------------
	# Plot MAE and Loss
	#------------------------------------------------
	plt.figure(figsize=(10, 6))
	plt.plot(epochs, mae, 'r')
	plt.plot(epochs, loss, 'b')
	plt.title('MAE and Loss')
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.legend(["MAE", "Loss"])
	plt.show()

	"""
	#------------------------------------------------
	# Plot Zoomed MAE and Loss
	#------------------------------------------------
	zoom = 50
	epochs_zoom = epochs[zoom:]
	mae_zoom = mae[zoom:]
	loss_zoom = loss[zoom:]


	plt.figure()
	plt.plot(epochs_zoom, mae_zoom, 'r')
	plt.plot(epochs_zoom, loss_zoom, 'b')
	plt.title('MAE and Loss')
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.legend(["MAE", "Loss"])

	plt.figure()
	plt.show()
	"""


#Main function ------------------------------------------------------------------------------------------


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('country_name', nargs="*", type=str)
	arguments = parser.parse_args()
	country_name = arguments.country_name
	country_name = '_'.join(country_name) #Make countryname from list to string separated by underscore


	time, cases, cases_scaler = setup_dataset(country_name)	


	#------------------------------------------------------------------------
	#Key variables
	#Current number of days is 390
	split_time = int(0.7*len(cases)) #70:30 split
	time_train = time[:split_time]
	x_train = cases[:split_time]
	time_valid = time[split_time:]
	x_valid = cases[split_time:]

	window_size = 5
	batch_size = len(cases) 
	shuffle_buffer_size = len(cases)
	epochs=200


	#------------------------------------------------------------------------
	#Function to determine the optimal learning rate (by inspection)
	dataset_lr = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
	learning_rate_optimal = learning_rate_optimizer(dataset_lr)

	#----------------------------------------------------------------------

	tf.keras.backend.clear_session() #must clear session as we defined variables for LR Scheduler

	#---------------------------------------------------------------------------------
	#NN with optimal Learning Rate
	#dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
	dataset_full_windowed = windowed_dataset(cases, window_size, batch_size, shuffle_buffer_size)
	history, model = run_model(dataset_full_windowed, learning_rate_optimal, epochs)

	model_filename = 'models_h5/' + country_name + '.h5' 
	model.save(model_filename)


	#-----------------------------------------------------------------------------------------
	#Predicting/forecasting

	forecast = model_forecast(model, cases[..., np.newaxis], window_size)
	results = forecast[split_time - window_size:-1, -1, 0]


	#Quantify the difference in predicted and actual values
	#mae_valid = tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
	#print(mae_valid)


	#basic_visualizations(time_valid, x_valid, results, history, epochs, cases_scaler)
	


main()
