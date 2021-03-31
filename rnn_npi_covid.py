import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
#from tensorflow import keras
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
	ts_greece = setup_ts(df, country_name)


	#Update timeseries so that the first value is the first confirmed case
	fnz = first_nonzero_case_index(ts_greece)
	ts_greece = ts_greece[fnz:-1] #Remove last row as it is just filled with NaN


	#2D numpy array (doesn't include date column within the array)
	dataset_full = ts_greece.to_numpy() #Remove last row as it is just filled with NaN
	#print(dataset)

	#Rescale the ConfirmedCases & ConfirmedDeaths which are the only features on the real number range
	#Create 2 different scalers as it makes it easier for later when plotting different graphs
	cases_scaler = MinMaxScaler()
	deaths_scaler = MinMaxScaler()

	#Inner reshape is to satisfy sklearn, outer reshape is to keep the shape of array the same as originally
	dataset_full[:,-2] = cases_scaler.fit_transform(dataset_full[:,-2].reshape(-1,1))[:,0]
	#dataset_full[:,-1] = deaths_scaler.fit_transform(dataset_full[:,-1].reshape(-1,1))[:,0]

	"""
	#Might be useful for when I bring in multidimensional data (as I will need to rescale both cases and deaths)

	#Original scaler which incoporated both cases and deaths (harder for visualizations)
	scaler = MinMaxScaler()
	dataset[:,-2:] = scaler.fit_transform(dataset[:,-2:])
	"""
	#Timeseries is useful for any initial visualizations, dataset is for the TF model


	time = np.arange(len(dataset_full), dtype="float32") #Time is represented as day x since first covid case
	cases = dataset_full[:,-2]
	#deaths = dataset_full[:,-1]

	return time, cases, cases_scaler 



def windowed_dataset(series, window_size, batch_size, shuffle_buffer, num_of_days_to_predict):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + num_of_days_to_predict, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + num_of_days_to_predict))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-num_of_days_to_predict], window[-num_of_days_to_predict:]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

def learning_rate_optimizer(x_train, window_size, batch_size, shuffle_buffer_size):
	dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

	#print(dataset)
	#for x, y in dataset:
	#   print(x, y)

	model = tf.keras.models.Sequential([
		 tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
			      input_shape=[None]),
		 tf.keras.layers.LSTM(64, return_sequences=True),
	#    	 tf.keras.layers.LSTM(64, return_sequences=True),
		 tf.keras.layers.Dense(output_size),
		#  tf.keras.layers.Lambda(lambda x: x * 100.0)
	])



	#-----------------------------------------------------------------------
	#Learning rate Scheduker
	lr_schedule = tf.keras.callbacks.LearningRateScheduler(
	    lambda epoch: 1e-8 * 10**(epoch / 20))
	optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
	model.compile(loss=tf.keras.losses.Huber(),
		      optimizer=optimizer,
		      metrics=["mae"])
	history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])

	plt.semilogx(history.history["lr"], history.history["loss"])
	#plt.axis([2e-8, 1e-4, 0, 30])
	plt.show()
	

def run_model(dataset, output_size, learning_rate_optimal, epochs):

	model = tf.keras.models.Sequential([
		 tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
			      input_shape=[None]),
		# tf.keras.layers.LSTM(64, return_sequences=True),
		# tf.keras.layers.LSTM(64, return_sequences=True),
		# tf.keras.layers.LSTM(64, return_sequences=True),
		 tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
		# tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
	#    	 tf.keras.layers.LSTM(32, return_sequences=True),
	#    	 tf.keras.layers.LSTM(32),
		 tf.keras.layers.Dense(output_size),
	])


	model.summary()

	#model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=learning_rate_optimal, momentum=0.9),metrics=["mae"])

	#Huber loss function version
	model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(lr=learning_rate_optimal, momentum=0.9),metrics=["mae"])

	#Compie the model with the entire database (as x_valid will have more recent data which is useful for predictions)
	history = model.fit(dataset, epochs=epochs)

	return history, model




def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast



def visualizations(time_valid, x_valid, results, history, epochs, cases_scaler):
	#For visualizations, we need to unscale the ConfirmedCases & ConfirmedDeaths
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





def main():
	country_name = sys.argv[1] 
	time, cases, cases_scaler = setup_dataset(country_name)	
	#print(cases)
	#------------------------------------------------------------------------
	#Key variables
	#Current number of days is 390
	split_time = int(0.7*len(cases)) #70:30 split
	time_train = time[:split_time]
	x_train = cases[:split_time]
	time_valid = time[split_time:]
	x_valid = cases[split_time:]

	window_size = 5
	batch_size = 20
	shuffle_buffer_size = 390
	output_size = 1 
	num_of_days_to_predict = 1
	epochs=100


	#------------------------------------------------------------------------
	#Function to determine the optimal learning rate (by inspection)
	#learning_rate_optimizer(x_train, window_size, batch_size, shuffle_buffer_size)
	learning_rate_optimal = 1e-3

	#----------------------------------------------------------------------

	tf.keras.backend.clear_session() #must clear session as we defined variables for LR Scheduler

	#---------------------------------------------------------------------------------
	#NN with optimal Learning Rate
	#dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
	dataset_full_windowed = windowed_dataset(cases, window_size, batch_size, shuffle_buffer_size, num_of_days_to_predict)
	history, model = run_model(dataset_full_windowed, output_size, learning_rate_optimal, epochs)

	model_filename = 'weights/model_' + country_name + '.h5' 
	model.save(model_filename)


	#-----------------------------------------------------------------------------------------
	#Predicting/forecasting

	forecast = model_forecast(model, cases[..., np.newaxis], window_size)
	results = forecast[split_time - window_size:-1, -1, 0]


	#Quantify the difference in predicted and actual values
	mae_valid = tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
	print(mae_valid)


	visualizations(time_valid, x_valid, results, history, epochs, cases_scaler)
	


main()
