from keras.models import Model
from keras.layers import Input,Lambda,GlobalAveragePooling2D
from keras.applications import ResNet50,InceptionV3,Xception,resnet50,inception_v3,xception
from keras.preprocessing.image import ImageDataGenerator
import h5py
#import densenet161

import tensorflow as tf
tf.test.gpu_device_name()
#weights_path = 'data/densenet161_weights_tf.h5'

def write_gap(MODEL,image_size,name,preprocess):
    width=image_size[0]
    height=image_size[1]
    input_tensor=Input((height,width,3))
    x = input_tensor
    x = Lambda(preprocess)(x)
    #base_model = densenet161.DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)
    base_model = MODEL(input_tensor=x,weights='imagenet',include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory('cut/', target_size=image_size, batch_size=1,shuffle=False)
    train=model.predict_generator(train_generator,train_generator.samples)

    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory('val1/', target_size=image_size, batch_size=1,shuffle=False)
    test=model.predict_generator(test_generator,test_generator.samples)
    h=h5py.File("gap/gap_"+name+"_cut1.h5")
    h.create_dataset("train",data=train)
    h.create_dataset("train_label",data=train_generator.classes)
    h.create_dataset("val",data=test)
    h.create_dataset("val_label",data=test_generator.classes)
    h.close()

write_gap(ResNet50,(224,224),'Resnet50',resnet50.preprocess_input)
write_gap(InceptionV3,(299,299),'InceptionV3',inception_v3.preprocess_input)
write_gap(Xception,(299,299),'Xception',xception.preprocess_input)
#write_gap(Xception,(224,224),densenet161.preprocess_input)