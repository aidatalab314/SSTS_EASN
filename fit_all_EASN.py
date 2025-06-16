import numpy as np
import pandas as pd
import os
import tensorflow.keras as keras

X_train = []
Y_train = []

X_test = []
Y_test = []

X_val = []
Y_val = []

station_1 = '06-07'
station_2 = '09-10'
station_3 = '10-11'
station_4 = '12-13'
station_5 = '13-14'
station_6 = '14-15'

preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_1+'/train/' 




for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_train.extend(temp)
        for j in range(temp.shape[0]):
            Y_train.append(clabel)

preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_2+'/train/' 

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_train.extend(temp)
        for j in range(temp.shape[0]):
            Y_train.append(clabel)

preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_3+'/train/' 

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_train.extend(temp)
        for j in range(temp.shape[0]):
            Y_train.append(clabel)

preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_4+'/train/' 

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_train.extend(temp)
        for j in range(temp.shape[0]):
            Y_train.append(clabel)

preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_5+'/train/' 

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_train.extend(temp)
        for j in range(temp.shape[0]):
            Y_train.append(clabel)

preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_6+'/train/' 

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_train.extend(temp)
        for j in range(temp.shape[0]):
            Y_train.append(clabel)
            
X_train = np.array(X_train)
Y_train = np.array(Y_train)



preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_1+'/test/'

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_test.extend(temp)
        for j in range(temp.shape[0]):
            Y_test.append(clabel)


preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_2+'/test/'

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_test.extend(temp)
        for j in range(temp.shape[0]):
            Y_test.append(clabel)

preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_3+'/test/'

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_test.extend(temp)
        for j in range(temp.shape[0]):
            Y_test.append(clabel)

preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_4+'/test/'

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_test.extend(temp)
        for j in range(temp.shape[0]):
            Y_test.append(clabel)

preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_5+'/test/'

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_test.extend(temp)
        for j in range(temp.shape[0]):
            Y_test.append(clabel)

preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_6+'/test/'

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_test.extend(temp)
        for j in range(temp.shape[0]):
            Y_test.append(clabel)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_1+'/val/'

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_val.extend(temp)
        for j in range(temp.shape[0]):
            Y_val.append(clabel)

preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_2+'/val/'

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_val.extend(temp)
        for j in range(temp.shape[0]):
            Y_val.append(clabel)

preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_3+'/val/'

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_val.extend(temp)
        for j in range(temp.shape[0]):
            Y_val.append(clabel)

preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_4+'/val/'

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_val.extend(temp)
        for j in range(temp.shape[0]):
            Y_val.append(clabel)

preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_5+'/val/'

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_val.extend(temp)
        for j in range(temp.shape[0]):
            Y_val.append(clabel)

preproc_dir = '/Users/jaysu/Desktop/PSSS-MCVT-exp/data/'+station_6+'/val/'

for i in os.listdir(preproc_dir):
    if 'id' not in i :
        label = i.split('_')
        # print(label)
        clabel = int(label[1])-1
        # print(clabel)
        temp = np.load(preproc_dir + i)
        X_val.extend(temp)
        for j in range(temp.shape[0]):
            Y_val.append(clabel)

X_val = np.array(X_val)
Y_val = np.array(Y_val)

#one-hot
Y_train = np.reshape(Y_train,(len(Y_train),1))
Y_train = np.eye(4, dtype='float32')[Y_train[:, 0]]
Y_test = np.reshape(Y_test,(len(Y_test),1))
Y_test = np.eye(4, dtype='float32')[Y_test[:, 0]]
Y_val = np.reshape(Y_val,(len(Y_val),1))
Y_val = np.eye(4, dtype='float32')[Y_val[:, 0]]

batch_size = 32
epochs = 30

#call back early stop
def call_list_fun(models, model_name):
    model_dir = './Model/{}-logs'.format(model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logfiles = model_dir + '/{}-{}'.format('basic_model',
                                           models.__class__.__name__)
    model_cbk = keras.callbacks.TensorBoard(log_dir=logfiles,
                                            histogram_freq=1)

    modelfiles = model_dir + '/{}-best-model.h5'.format('basic_model')
    model_mckp = keras.callbacks.ModelCheckpoint(modelfiles,
                                                 monitor='val_accuracy',
                                                 save_best_only=True)

    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=30,
                                              verbose=2)
    return [model_cbk, model_mckp, earlystop]

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
import seaborn as sns

import tensorflow as tf

from keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Lambda, DepthwiseConv2D, Add, Reshape, Permute, concatenate, Average, AveragePooling2D, Flatten, multiply
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


def get_flops(model, batch_size=None):
    if batch_size is None:
        batch_size = 1

    real_model = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + model.inputs[0].shape[1:], model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                            run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops


def SPM(x, groups):

    # Channel split
    in_channels = x.shape[-1]
    channel_per_group = in_channels // groups
    group_list = []

    group_list.append(x)

    for group in range(groups):
        group_start = group * channel_per_group
        group_end = (group + 1) * channel_per_group
        x_group = x[:, :, :, group_start:group_end]
        print('x_group shape',x_group.shape)

        # Multiscale Depthwise Convolution
        x_group = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(x_group)
        x_group = BatchNormalization()(x_group)
        x_group = Activation('relu')(x_group)

        x_group = Conv2D(filters=channel_per_group, kernel_size=(1, 1), strides=(1, 1), padding='same')(x_group)
        x_group = BatchNormalization()(x_group)
        x_group = Activation('relu')(x_group)

        group_list.append(x_group)
        print(len(group_list))

    x = concatenate(group_list)

    # Shuffle
    _, h, w, ch = x.shape
    print('x shape',x.shape)
    x = Reshape((h, w, groups, ch // groups))(x)
    print('x first reshape',x.shape)
    x = Permute((1, 2, 4, 3))(x)
    print('x permute',x.shape)
    x = Reshape((h, w, ch))(x)
    print('x second reshape',x.shape)

    return x


def SAM(x):

    x1 = x[:, :, :, :x.shape[3]//2]
    x2 = x[:, :, :, -x.shape[3]//2:]


    # channel
    squeeze_ch = GlobalAveragePooling2D()(x1)
    squeeze_ch = Reshape((1, 1, -1))(squeeze_ch)
    excitation_ch = Dense(units=x1.shape[-1] // 2, activation='relu')(squeeze_ch) 
    excitation_ch = Dense(units=x1.shape[-1], activation='sigmoid')(excitation_ch) 
    excitation_ch = Reshape((1, 1, -1))(excitation_ch)
    x1 = multiply([x1, excitation_ch])


    # spatial
    squeeze_sp = BatchNormalization()(x2) 
    print("Spatial att step 1", squeeze_sp.shape)
    excitation_sp = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(squeeze_sp)
    print("Spatial att step 2", excitation_sp.shape)
    x2 = multiply([x2, excitation_sp])
    print("Spatial att ouptput", excitation_sp.shape)

    x = concatenate([x1,x2])

    return x


def EASN(input_shape, num_classes, groups=3):

    input_tensor = Input(shape=input_shape)

    # Initial Convolution
    # x = Conv2D(filters=24, kernel_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
    x = Conv2D(filters=24, kernel_size=(3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # ShuffleNet blocks
    x = SPM(x, groups)
    x = SAM(x)
    print('x first block shape',x.shape)

    x = SPM(x, groups)
    x = SAM(x)
    print('x second block shape',x.shape)

    x = SPM(x, groups)
    x = SAM(x)
    print('x third block shape',x.shape)

    x = SPM(x, groups)
    x = SAM(x)
    print('x fourth block shape',x.shape)

    x = SPM(x, groups)
    x = SAM(x)
    print('x fifth block shape',x.shape)

    # x = SPM(x, groups)
    # x = SAM(x)
    # print('x sixth block shape',x.shape)

    # x = SPM(x, groups)
    # x = SAM(x)
    # print('x seventh block shape',x.shape)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=x)

    return model


input_shape = (64, 64, 3)
num_classes = 4

model = EASN(input_shape, num_classes)
model.summary()
plot_model(model, to_file='img.png', show_shapes=True)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
callbacks_list = call_list_fun(model, 'cnn_model')

learning_rate = 1e-4
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# Model training
model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

cnn_history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=30,
                        validation_split=0.2,
                        shuffle = True,
                        verbose=1,
                        callbacks=callbacks_list)

model.save('easn_model.h5')
model.save('easn_saved_model', save_format='tf')

y_pred = model.predict(X_val)
y_pred = y_pred.argmax(-1)
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score
print('accuracy : ',accuracy_score(Y_val.argmax(-1), y_pred))
print('confusion matrix : \n',confusion_matrix(Y_val.argmax(-1), y_pred)) 
print('f1_score : \n',f1_score(Y_val.argmax(-1),y_pred,average=None))

flops = get_flops(model, 1)
print(f"FLOPS: {flops}")