import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array,load_img

datagen = ImageDataGenerator(rotation_range=40,
							 width_shift_range=0.2,
							 height_shift_range=0.2,
							 shear_range=0.2,
							 zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')


origin = "data/cut/"
add = "data/add_cut/"


for dir in os.listdir(origin):
	print(dir)
	for filename in os.listdir(origin+dir):
		if os.path.exists(add+dir)==False:
			os.mkdir(add+dir)
		img = load_img(origin+dir+"/"+filename)
		x = img_to_array(img)
		x = x.reshape((1,) + x.shape)
		i=0
		for batch in datagen.flow(x, batch_size=1,save_to_dir=add+dir+"/", save_prefix='dog', save_format='jpg'):
			i += 1
			if i > 0:
				break