import h5py
import numpy as np
from sklearn.utils import shuffle
import keras
from keras.models import *
from keras.layers import *
np.random.seed(2017)
X_train=[]
X_test=[]
X_valid=[]

from sklearn.datasets import load_files
from keras.utils import np_utils

# 定义函数来加载train，test和validation数据集
def load_dataset(path):
    data = load_files(path)
    #dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_targets

# 加载train，test和validation数据集
train_targets = load_dataset('data/help/dogImages/train.bak')
valid_targets = load_dataset('data/help/dogImages/valid')

#,"Xception"]:
train = []
valid = []
input_ = []
x = []
i = 0

def newModel(path):
    train.append(np.load(path)['train'])
    k=0
    j=0
    for feature in train[0]:
        a=feature[0][0][0]
        if abs(a- 2.5634682)<0.0001:#0.14260143:
            j = k
            break
        k += 1
    valid.append(np.load(path)['valid'])

    input_.append(Input(shape=train[i].shape[1:]))
    x.append(GlobalAveragePooling2D(input_shape=train[i].shape[1:])(input_[i]))
    x[i] = Dense(133, activation='softmax')(x[i])

    global i
    i = i+ 1

newModel('./data/help/bottleneck/DogResnet50Data.npz')
# newModel('./data/help/bottleneck/DogInceptionV3Data.npz')
#newModel('./data/help/bottleneck/DogXceptionData.npz')

#x = Concatenate()(x)
#x = Average()(x)
out = x[0]#Dense(133, activation='softmax')(x[0])
model = Model(inputs=input_[0], outputs=out)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
checkpointer = keras.callbacks.ModelCheckpoint(filepath='output/weights.best.help1.hdf5',
                               verbose=1, save_best_only=True)

model.load_weights('output/weights.best.help1.hdf5')
# model.fit(train[0],train_targets, validation_data=(valid[0], valid_targets),
#            batch_size=128,epochs = 20,callbacks=[checkpointer],validation_split=0.2)

#
# X_test=[]
# test_targets = load_dataset('data/dogImages/test')
#
# for filename in ["Resnet50","Xception","InceptionV3"]:
#     bottleneck_features = np.load('./data/bottleneck/Dog'+filename+'Data.npz')
#     X_test = np.append(X_train,np.array(bottleneck_features['test']))
#
# X_test = np.concatenate(X_test, axis=1)
#
# # 获取测试数据集中每一个图像所预测的狗品种的index
# VGG16_predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in X_test]
#
# # 报告测试准确率
# test_accuracy = 100 * np.sum(np.array(VGG16_predictions) == np.argmax(test_targets, axis=1)) / len(VGG16_predictions)
# print('Test accuracy: %.4f%%' % test_accuracy)

from extract_bottle_features import *

img_path = "data/train/0/903974728,3645935065.jpg"
bottleneck_feature = []
bottleneck_feature.append(extract_Resnet50(img_path))
# bottleneck_feature.append(extract_InceptionV3(img_path))
# bottleneck_feature.append(extract_Xception(img_path))
#bottleneck_feature1 = np.concatenate(bottleneck_feature, axis=1)

from glob import glob
dog_names = [item[20:-1] for item in sorted(glob("data/help/dogImages/train.bak/*/"))]
predicted_vector = model.predict(bottleneck_feature[0])
print(dog_names[np.argmax(predicted_vector)])
predict = np.argmax(predicted_vector)
print(predict)