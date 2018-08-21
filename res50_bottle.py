from extract_bottle_features import *
import h5py


#img_path = "data/help/dogImages/train1/001.Affenpinscher/Affenpinscher_00001.jpg"
img_path = "data/train/0/82161122,3339619667.jpg"

v = extract_Resnet50(img_path)

h = h5py.File("output/gap_InceptionV3_train_nopool1.h5")
