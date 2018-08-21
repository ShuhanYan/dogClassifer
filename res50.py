from keras.models import Model
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50
import dataLoader as dl
import keras
from keras import optimizers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_classes = 100
epochs = 1

# 训练图像的数量
image_numbers = dl.train_generator.samples

# 使用ResNet的结构，不包括最后一层，且加载ImageNet的预训练参数
base_model = ResNet50(weights = "resnet50.h5'", include_top = False, pooling = 'avg')

for layer in base_model.layers[:-3]:
    layer.trainable = False
for layer in base_model.layers[-3:]:
    layer.trainable = True

predictions = Dense(num_classes, activation='softmax')(base_model.output)

# 定义整个模型
model = Model(inputs=base_model.input, outputs=predictions)
#model.summary()

# 编译模型，loss为交叉熵损失
model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001,momentum=0.7,nesterov=True), metrics=["accuracy"])
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True)
checkpoint = keras.callbacks.ModelCheckpoint("resnet50.h5", monitor='val_acc', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

# 训练模型
model.fit_generator(dl.train_generator,steps_per_epoch = 200, epochs = epochs,
                    validation_data = dl.validation_generator, validation_steps = dl.batch_size, callbacks=callbacks_list)