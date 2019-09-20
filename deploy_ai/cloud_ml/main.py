from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

model = InceptionV3()

def load_image(image_path):
	img = image.load_img(image_path, target_size=(128, 128))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

image = load_image('elephant.jpg')
preds = model.predict(image)

predictions = decode_predictions(preds, top=3)[0]

print('Predicted:', )
for _, _class, score in predictions:
	print(score, _class)
