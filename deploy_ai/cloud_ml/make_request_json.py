from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.preprocessing import image
import json
import numpy as np

def load_image(image_path):
	img = image.load_img(image_path, target_size=(128, 128))
	x = image.img_to_array(img)
	# x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

img = load_image('elephant.jpg')

with open('request.json', 'wt') as f:
	json.dump({"input_1": img.tolist()}, f)
