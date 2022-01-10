# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 22:46:01 2022

@author: Maysam
"""


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from spektral.datasets import TUDataset
from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.transforms import AdjToSpTensor, LayerPreprocess
from load_data import read_data
import numpy as np

from GWCN_Net import AGWConv
from GWCN_Func import TimeHistory,computeLoaderCombine
time_callback = TimeHistory()



# Load data ============================================================================

# load data parameters
dataset_name="G4.mat"
path="dataset"
dtype=np.float32    # `dtype`: numpy dtype of graph data
ToSpTensor=True
modalities=[1]
N_modalities=len(modalities)
split_range=[0.5,0.3,0.2] #train/validation/test split

dataset, mask_tr, mask_va, mask_te=read_data(
    dataset_name,path,modalities, split_range,dtype,ToSpTensor)


# Parameters
channels = 16  # Number of features in the first layer
iterations =1  # Number of layers
share_weights = False  # Share weights 
dropout_skip = 0.75  # Dropout rate for the internal skip connection 
dropout = 0.25  # Dropout rate for the features
l2_reg = 5e-4  # L2 regularization rate
learning_rate = 1e-2  # Learning rate
epochs = 20000  # Number of training epochs
patience = 100  # Patience for early stopping
a_dtype = dataset.a.dtype  # Only needed for TF 2.1

N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels  # Number of classes

# AGWC parameters
thr=1e-4    # threshold parameter for check sparsity of wavelet
scales=[0.4]      # range of scales  [0.4,0.9] 
N_scales = len(scales)
m=40                #Order of polynomial approximation
apx_phsi= False     # approximate Phsi


# Model definition
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True, dtype=a_dtype)
phsi_in = Input(shape=(N,N,), dtype=a_dtype)
phsiIn_in = Input(shape=(N,N,), dtype=a_dtype)




gc_1 = AGWConv(
    channels,
    iterations=iterations,
    order=N_scales,
    share_weights=share_weights,
    dropout_rate=dropout_skip,
    activation="elu",
    gcn_activation="elu",
    kernel_regularizer=l2(l2_reg),
)([x_in, phsi_in, phsiIn_in, a_in])
gc_2 = Dropout(dropout)(gc_1)
gc_2 = AGWConv(
    n_out,
    iterations=1,
    order=1,
    share_weights=share_weights,
    dropout_rate=dropout_skip,
    activation="softmax",
    gcn_activation=None,
    kernel_regularizer=l2(l2_reg),
)([gc_2, phsi_in, phsiIn_in, a_in])


# Build model
model = Model(inputs=[x_in, phsi_in, phsiIn_in, a_in], outputs=gc_2)
optimizer = Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", weighted_metrics=["acc"]
)
model.summary()


# Build data loader ======================================================================

loader_tr_load,loader_va_load,loader_te_load=computeLoaderCombine(
    dataset,mask_tr,mask_va,mask_te,apx_phsi,N_scales,scales,m,epochs,thr)




print("Training model.")
history=model.fit(
    loader_tr_load,
    steps_per_epoch=1,
    validation_data=loader_va_load,
    validation_steps=1,
    epochs=epochs,
    callbacks=[EarlyStopping(patience=patience, restore_best_weights=True),time_callback],
)

times = time_callback.times

model.summary()

model.save('model.h5')
model.save_weights('model_weights.h5')

# Evaluate model
print("Evaluating model.")


eval_results = model.evaluate(loader_te_load, steps=1)
print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))

print("\n""Avg. Time/epoch: {}\n" .format(np.average(times)))

# import stellargraph as sg
# sg.utils.plot_history(history)

# get parameters 
ww=model.get_weights()