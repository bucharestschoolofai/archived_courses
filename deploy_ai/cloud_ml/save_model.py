from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import tensorflow as tf

model = InceptionV3()
saved_to_path = tf.contrib.saved_model.save_keras_model(model, 'inception_saved_model/')
