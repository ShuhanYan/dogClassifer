import h5py
import numpy as np
from sklearn.utils import shuffle
import keras
from keras.models import *
from keras.layers import *

np.random.seed(2018)
from keras.utils import np_utils
from keras import optimizers

train = []
test = []
input_ = []
train_label = []
test_label = []
x = []

with h5py.File('output/gap_InceptionV3_train.h5', 'r') as h:
    train = (np.array(h['train']))
with h5py.File('data/gap/gap_InceptionV3_train.h5', 'r') as h:
    train_label = np.array(h['train_label'])

train,y = shuffle(train,train_label)

train_label = np_utils.to_categorical(y, 100)
#     with h5py.File("drive/dog/gap_"+path+'_test.h5', 'r') as h2:
#       test.append(np.array(h2['test']))

#      test_label=h2['test_label']

i = 0
input_= (Input(shape=train.shape[1:]))
#x.append(GlobalAveragePooling2D(input_shape=train.shape[1:])(input_[i]))
# x[i] = Dense(512, activation='relu')(x[i])
# x[i] = Dropout(0.5)(x[i])
# x[i] = Dense(100, activation='softmax')(x[i])

# i = 1
# input_.append(Input(shape=train[i].shape[1:]))
# x.append(GlobalAveragePooling2D(input_shape=train[i].shape[1:])(input_[i]))
# x[i] = Dense(512, activation='relu')(x[i])
# x[i] = Dropout(0.5)(x[i])
# x[i] = Dense(100, activation='softmax')(x[i])

#x = Average()(x)
#x = Concatenate()(x)
x = Dense(512,activation='relu')(input_)
x = Dropout(0.5)(x)
x = Dense(100,activation='softmax')(x)
out = x  # Dense(100, activation='softmax')(x[0])
model = Model(inputs=input_, outputs=out)
#model.load_weights('output/weights.best.merge.hdf5')

# model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=['accuracy'])
checkpointer = keras.callbacks.ModelCheckpoint(filepath='output/weights.best.mm.hdf5',
                                                verbose=1, save_best_only=True)
#
# model.fit(train, train_label,
#           batch_size=32, epochs=20, validation_split=0.02, callbacks=[checkpointer])

# model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(train, train_label, batch_size=64,epochs=12, validation_split=0.0)

keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0001,nesterov=True)
model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train, train_label, batch_size=32,epochs=20, validation_split=0.2,callbacks=[checkpointer])

keras.optimizers.SGD(lr=0.001, momentum=0.9,decay=0.0001,nesterov=True)
model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train, train_label, batch_size=128,epochs=10, validation_split=0.2,callbacks=[checkpointer])

keras.optimizers.SGD(lr=0.0001, momentum=0.9,decay=0.0001,nesterov=True)
model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train, train_label, batch_size=128,epochs=20, validation_split=0.2,callbacks=[checkpointer])

#model.save_weights('output/weights.merge.hdf5')