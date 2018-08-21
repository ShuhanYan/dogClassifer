from keras.models import *
from keras.layers import *
from keras.preprocessing.image import *
import h5py
import densenet161
from keras import backend as K

weights_path = '/your path/wh_code/densenet/densenet161_weights_tf.h5'
image_size = (224, 224)
base_model = densenet161.DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)
model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
gen = ImageDataGenerator(preprocessing_function=densenet161.preprocess_input)
test_generator = gen.flow_from_directory("/your path/test", image_size, shuffle=False, batch_size=8,
                                         class_mode=None)
train_generator = gen.flow_from_directory("/your path/train3p", image_size, shuffle=False, batch_size=8)
train = model.predict_generator(train_generator, 6910, verbose=True)
test = model.predict_generator(test_generator, 1325, verbose=True)

# print(test .shape)
with h5py.File(fileDir + "wh_code/nocut/224_3p-dense161.h5") as h:
    h.create_dataset("train", data=train)
    h.create_dataset("test", data=test)
    h.create_dataset("label", data=train_generator.classes)
    # h.create_dataset("val", data=val)
    # h.create_dataset("trainlabel", data=train_generator.classes)
    # h.create_dataset("vallabel", data=val_generator.classes)

K.clear_session()