import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import tensorflow as tf
#from tensorflow import keras
from math import isnan



#Creates the timeseries so that it can be handled by TF
def setup_ts(dataframe, country_name):
	df_country = df.loc[df['CountryName'] == country_name]
	df_country.loc[:,'Date'] = pd.to_datetime(df_country.loc[:, 'Date'], format='%Y%m%d')
	#df_greece['Date'] = pd.to_datetime(df_greece['Date'], format='%Y%m%d')
	ts_country = df_country.set_index('Date')
	ts_country = ts_country.drop(columns=['CountryName', 'CountryCode'])
	return ts_country

#Finds the first non-zero confirmed case
def first_nonzero_case_index(timeseries):
	confirmed_cases = ts_greece['ConfirmedCases'].to_numpy()
	fnz = 0 
	for i in confirmed_cases:
		if isnan(i) == False and i != 0:
			break
		fnz += 1	
	return fnz




df = pd.read_csv('Data/OxCGRT_latest_cleaned.csv', index_col='Index')

#Convert the csv into an appropriate timeseries
ts_greece = setup_ts(df, 'Greece')

#Update timeseries so that the first value is the first confirmed case
fnz = first_nonzero_case_index(ts_greece)
ts_greece = ts_greece[fnz:]
print(ts_greece)

#2D numpy array (doesn't include date column within the array)
dataset = ts_greece.to_numpy()[:-1,:] #Remove last row as it is just filled with NaN
print(dataset)






"""
#------------------------------------------------------------------------
#Key variables

#Series is all the data we have 
#time = np.arange(4 * 365 + 1, dtype="float32")

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


#Alternative to keep the code clean if it gets too large
#(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

#------------------------------------------------------------------------

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

tf.keras.backend.clear_session()


dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)




model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
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
plt.axis([1e-8, 1e-4, 0, 30])


learning_rate_optimal = 1e-3

#----------------------------------------------------------------------



tf.keras.backend.clear_session() #must clear session as we defined variables for LR Scheduler



#---------------------------------------------------------------------------------
#NN with optimal Learning Rate
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])


model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=learning_rate_optimal, momentum=0.9),metrics=["mae"])

#Huber loss function version
model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(lr=learning_rate_optimal, momentum=0.9),metrics=["mae"])


history = model.fit(dataset, epochs=100)

#-----------------------------------------------------------------------------------------
#Predicting/forecasting
"""
"""
#Alternative to the below
forecast = []
results = []
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]
"""
"""
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]



plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)

tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()


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
plt.plot(epochs, mae, 'r')
plt.plot(epochs, loss, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.figure()


#------------------------------------------------
# Plot Zoomed MAE and Loss
#------------------------------------------------

epochs_zoom = epochs[200:]
mae_zoom = mae[200:]
loss_zoom = loss[200:]


plt.plot(epochs_zoom, mae_zoom, 'r')
plt.plot(epochs_zoom, loss_zoom, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.figure()

"""
