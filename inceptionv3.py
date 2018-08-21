from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
import keras
from keras import optimizers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_classes = 100
epochs = 8
batch_size = 32

from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


# 从文件中读取数据，目录结构应为train下面是各个类别的子目录，每个子目录中为对应类别的图像
train_generator = train_datagen.flow_from_directory('./data/train/', target_size = (299, 299), batch_size = batch_size)

# 生成测试数据
test_datagen = ImageDataGenerator()
validation_generator = test_datagen.flow_from_directory('./data/test1/', target_size = (299, 299), batch_size = batch_size)


image_numbers = train_generator.samples

base_model = InceptionV3(include_top=False)

for layer in base_model.layers[:172]:
    layer.trainable = False
for layer in base_model.layers[172:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x) #new FC layer, random init

predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
#model.summary()
model.load_weights('resnet50.h5')

model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001,momentum=0.7,nesterov=True), metrics=["accuracy"])
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True)
checkpoint = keras.callbacks.ModelCheckpoint("inceptionv3.h5", monitor='val_acc', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

# 训练模型
model.fit_generator(train_generator,steps_per_epoch = 200, epochs = epochs,
                    validation_data = validation_generator, validation_steps = batch_size, callbacks=callbacks_list)