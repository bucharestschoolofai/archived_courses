from keras.applications.inception_v3 import decode_predictions
import numpy as np

preds = []

predictions = decode_predictions(np.array(preds).reshape(1, -1), top=3)[0]

print('Predicted:', )
for _, _class, score in predictions:
	print(score, _class)
