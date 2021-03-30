import h5py
import numpy as np
filename = "my_model.h5"

f = h5py.File(filename, 'r')
keys = f.keys()
print("Keys: ", list(keys))



optimizer_weights = f['model_weights']
print(optimizer_weights)
print(list(optimizer_weights))

"""
#Model Weights

#Dense layer
dense_bias = f['model_weights/dense/dense/bias:0']
print(dense_bias.shape)

dense_kernel = f['model_weights/dense/dense/kernel:0']
print(dense_kernel.shape)
print(np.array(dense_kernel))

#Backward LSTM
backward_lstm_cell_2_bias = f['model_weights/bidirectional/bidirectional/backward_lstm/lstm_cell_2/bias:0']
print(backward_lstm_cell_2_bias.shape)

backward_lstm_cell_2_kernel = f['model_weights/bidirectional/bidirectional/backward_lstm/lstm_cell_2/kernel:0']
print(backward_lstm_cell_2_kernel.shape)

backward_lstm_cell_2_recurrent_kernel = f['model_weights/bidirectional/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel:0']
print(backward_lstm_cell_2_recurrent_kernel.shape)

#Forward LSTM
forward_lstm_cell_1_bias = f['model_weights/bidirectional/bidirectional/forward_lstm/lstm_cell_1/bias:0']
print(forward_lstm_cell_1_bias.shape)

forward_lstm_cell_1_kernel = f['model_weights/bidirectional/bidirectional/forward_lstm/lstm_cell_1/kernel:0']
print(forward_lstm_cell_1_kernel.shape)

forward_lstm_cell_1_recurrent_kernel = f['model_weights/bidirectional/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel:0']
print(forward_lstm_cell_1_recurrent_kernel.shape)

"""










"""
#Optimizer weights

#Backward LSTM

optimizer_weights_sgd = f['optimizer_weights/SGD/bidirectional/backward_lstm/lstm_cell_2/bias/momentum:0']
print(optimizer_weights_sgd)
print(list(optimizer_weights_sgd))


optimizer_weights_sgd = f['optimizer_weights/SGD/bidirectional/backward_lstm/lstm_cell_2/kernel/momentum:0']
print(optimizer_weights_sgd)
print(list(optimizer_weights_sgd))

optimizer_weights_sgd = f['optimizer_weights/SGD/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/momentum:0']
print(optimizer_weights_sgd)
print(list(optimizer_weights_sgd))



#Forward LSTM

optimizer_weights_sgd = f['optimizer_weights/SGD/bidirectional/forward_lstm/lstm_cell_1/bias/momentum:0']
print(optimizer_weights_sgd)
print(list(optimizer_weights_sgd))


optimizer_weights_sgd = f['optimizer_weights/SGD/bidirectional/forward_lstm/lstm_cell_1/kernel/momentum:0']
print(optimizer_weights_sgd)
print(list(optimizer_weights_sgd))


optimizer_weights_sgd = f['optimizer_weights/SGD/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/momentum:0']
print(optimizer_weights_sgd)
print(list(optimizer_weights_sgd))


optimizer_weights_kernel_momentum = f['optimizer_weights/SGD/dense/kernel/momentum:0']
print(optimizer_weights_kernel_momentum)
print(np.array(optimizer_weights_kernel_momentum))



optimizer_weights_bias_momentum = f['optimizer_weights/SGD/dense/bias/momentum:0']
print(optimizer_weights_bias_momentum)
print(np.array(optimizer_weights_bias_momentum))



iter_values = f['optimizer_weights/SGD/iter:0']
print(iter_values)
print(iter_values.shape)
print(np.array(iter_values))


"""

