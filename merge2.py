import h5py
import numpy as np
from sklearn.utils import shuffle
import keras
from keras.models import *
from keras.layers import *
#from extract_bottle_features import *

np.random.seed(2018)
from keras.utils import np_utils
from keras import optimizers

train = []
train_label = []
train_add = []
train_label_add = []
val_label = []
x = []
y = []
val = []

with h5py.File('data/gap/gap_Resnet50_cut.h5', 'r') as h:
    train.append(np.array(h['train']))
    train_label = np.array(h['train_label'])
    val.append(np.array(h['val']))
    val_label = np.array(h['val_label'])

for filename in ['InceptionV3','Xception']:
    with h5py.File('data/gap/gap_'+filename+'_cut.h5', 'r') as h:
        train.append(np.array(h['train']))
        val.append(np.array(h['val']))

for filename in ['Resnet50','InceptionV3','Xception']:
    with h5py.File('data/gap/gap_'+filename+'_cut_add.h5', 'r') as h:
        train_add.append(np.array(h['train']))
        train_label_add = np.array(h['train_label'])


train_label = np_utils.to_categorical(train_label, 100)
train_label_add = np_utils.to_categorical(train_label_add, 100)
val_label = np_utils.to_categorical(val_label, 100)

train_add = np.concatenate(train_add, axis=1)
train = np.concatenate(train, axis=1)
val = np.concatenate(val, axis=1)

input_tensor = Input(train.shape[1:])
x = Dense(512,activation='relu')(input_tensor)
x = Dropout(0.5)(x)
x = Dense(100,activation='softmax')(x)
model = Model(input_tensor,x)
mode = ''
mode = "train"
if mode == "train":
    train = train
    train_label = train_label

    model.load_weights('output/weights.best.cut_add.hdf5')
    checkpointer = keras.callbacks.ModelCheckpoint(monitor='val_acc',filepath='output/weights.best.cut_add.hdf5',
                                                    verbose=1, save_best_only=True)

    model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(train, train_label, batch_size=64,epochs=12, validation_data = (val,val_label), callbacks=[checkpointer])

    keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0001, nesterov=True)
    model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(train, train_label, batch_size=32,epochs=20, validation_data = (val,val_label), callbacks=[checkpointer])

    keras.optimizers.SGD(lr=0.001, momentum=0.9,decay=0.0001,nesterov=True)
    model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(train, train_label, batch_size=128,epochs=10, validation_data = (val,val_label),callbacks=[checkpointer])

    keras.optimizers.SGD(lr=0.0001, momentum=0.9,decay=0.0001,nesterov=True)
    model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(train, train_label, batch_size=128,epochs=20, validation_data = (val,val_label),callbacks=[checkpointer])

    # model.save_weights('output/weights.merge.hdf5')



mode = "val"

if mode == "val":
    model.load_weights('output/weights.best.cut_add.hdf5')

    predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in val]
    test_accuracy = 100 * np.sum(np.array(predictions) == np.argmax(val_label, axis=1)) / len(predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)

    # img_path = "data/train/0/301377872,168861123.jpg"
    # bottleneck_feature = []
    # bottleneck_feature.append(extract_InceptionV3(path_to_tensor(img_path)))
    # bottleneck_feature.append(extract_Resnet50(path_to_tensor(img_path)))
    # bottleneck_feature.append(extract_Xception(path_to_tensor(img_path)))
    # bottleneck_feature1 = np.concatenate(bottleneck_feature, axis=1)
    #
    # predicted_vector = model.predict(bottleneck_feature1)
    # predict = np.argmax(predicted_vector)
    # print(predict)
