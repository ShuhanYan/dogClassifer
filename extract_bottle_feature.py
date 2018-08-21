from keras.models import Model
from keras.layers import Input,Lambda,GlobalAveragePooling2D
from keras.preprocessing import image
import numpy as np

def extract_VGG16(img_path):
    tensor = path_to_tensor(img_path, (299, 299))
    from keras.applications.vgg16 import VGG16, preprocess_input
    base_model = VGG16(weights='imagenet', include_top=False)
    return Model(base_model.input, GlobalAveragePooling2D()(base_model.output)).predict(preprocess_input(tensor))

def extract_VGG19(tensor):
    from keras.applications.vgg19 import VGG19, preprocess_input
    return VGG19(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_Resnet50(img_path):
    tensor = path_to_tensor(img_path, (224, 224))

    from keras.applications.resnet50 import ResNet50, preprocess_input
    model = ResNet50(weights='imagenet', include_top=False)
    model = Model(model.input, GlobalAveragePooling2D()(model.output))
    return model.predict(preprocess_input(tensor))

def extract_Xception(img_path):
    tensor = path_to_tensor(img_path, (299, 299))

    from keras.applications.xception import Xception, preprocess_input
    model = Xception(weights='imagenet', include_top=False)
  #  model = Model(model.input, GlobalAveragePooling2D()(model.output))
    return model.predict(preprocess_input(tensor))

def extract_InceptionV3(img_path):
    tensor = path_to_tensor(img_path, (299, 299))

    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    model = InceptionV3(weights='imagenet', include_top=False)
   # model = Model(model.input, GlobalAveragePooling2D()(model.output))
    return model.predict(preprocess_input(tensor))


def path_to_tensor(img_path,size):
    img = image.load_img(img_path, target_size=size)
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)