import h5py
import numpy as np
filename = "weights/my_model.h5"

f = h5py.File(filename, 'r')
#keys = f.keys()
#print("Keys: ", list(keys))




#Dense layer
dense_bias = f['model_weights/dense/dense/bias:0']
dense_bias_array = np.array(dense_bias)
np.savetxt('weights/dense_bias.txt', dense_bias_array)

dense_kernel = f['model_weights/dense/dense/kernel:0']
dense_kernel_array = np.array(dense_kernel)
np.savetxt('weights/dense_kernel.txt', dense_kernel_array)



#Backward LSTM
backward_lstm_cell_2_bias = f['model_weights/bidirectional/bidirectional/backward_lstm/lstm_cell_2/bias:0']
backward_lstm_cell_2_bias_array = np.array(backward_lstm_cell_2_bias)
np.savetxt('weights/backward_bias.txt', backward_lstm_cell_2_bias_array)


backward_lstm_cell_2_kernel = f['model_weights/bidirectional/bidirectional/backward_lstm/lstm_cell_2/kernel:0']
backward_lstm_cell_2_kernel_array = np.array(backward_lstm_cell_2_kernel)
np.savetxt('weights/backward_kernel.txt', backward_lstm_cell_2_kernel_array)


backward_lstm_cell_2_recurrent_kernel = f['model_weights/bidirectional/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel:0']
backward_lstm_cell_2_recurrent_kernel_array = np.array(backward_lstm_cell_2_recurrent_kernel)
np.savetxt('weights/backward_recurrent_kernel.txt', backward_lstm_cell_2_recurrent_kernel_array)




#Forward LSTM
forward_lstm_cell_1_bias = f['model_weights/bidirectional/bidirectional/forward_lstm/lstm_cell_1/bias:0']
forward_lstm_cell_1_bias_array = np.array(forward_lstm_cell_1_bias)
np.savetxt('weights/forward_bias.txt', forward_lstm_cell_1_bias_array)


forward_lstm_cell_1_kernel = f['model_weights/bidirectional/bidirectional/forward_lstm/lstm_cell_1/kernel:0']
forward_lstm_cell_1_kernel_array = np.array(forward_lstm_cell_1_kernel)
np.savetxt('weights/forward_kernel.txt', forward_lstm_cell_1_kernel_array)



forward_lstm_cell_1_recurrent_kernel = f['model_weights/bidirectional/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel:0']
forward_lstm_cell_1_recurrent_kernel_array = np.array(forward_lstm_cell_1_recurrent_kernel)
np.savetxt('weights/forward_recurrent_kernel.txt', forward_lstm_cell_1_recurrent_kernel_array)







"""
#Optimizer weights

#Backward LSTM

optimizer_weights_sgd = f['optimizer_weights/SGD/bidirectional/backward_lstm/lstm_cell_2/bias/momentum:0']
print(optimizer_weights_sgd)
#print(list(optimizer_weights_sgd))


optimizer_weights_sgd = f['optimizer_weights/SGD/bidirectional/backward_lstm/lstm_cell_2/kernel/momentum:0']
print(optimizer_weights_sgd)
#print(list(optimizer_weights_sgd))

optimizer_weights_sgd = f['optimizer_weights/SGD/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/momentum:0']
print(optimizer_weights_sgd)
#print(list(optimizer_weights_sgd))



#Forward LSTM

optimizer_weights_sgd = f['optimizer_weights/SGD/bidirectional/forward_lstm/lstm_cell_1/bias/momentum:0']
print(optimizer_weights_sgd)
#print(list(optimizer_weights_sgd))


optimizer_weights_sgd = f['optimizer_weights/SGD/bidirectional/forward_lstm/lstm_cell_1/kernel/momentum:0']
print(optimizer_weights_sgd)
#print(list(optimizer_weights_sgd))


optimizer_weights_sgd = f['optimizer_weights/SGD/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/momentum:0']
print(optimizer_weights_sgd)
#print(list(optimizer_weights_sgd))


optimizer_weights_kernel_momentum = f['optimizer_weights/SGD/dense/kernel/momentum:0']
print(optimizer_weights_kernel_momentum)
#print(np.array(optimizer_weights_kernel_momentum))



optimizer_weights_bias_momentum = f['optimizer_weights/SGD/dense/bias/momentum:0']
print(optimizer_weights_bias_momentum)
#print(np.array(optimizer_weights_bias_momentum))



iter_values = f['optimizer_weights/SGD/iter:0']
print(iter_values)
print(iter_values.shape)
#print(np.array(iter_values))

"""
