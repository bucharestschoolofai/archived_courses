import tensorflow as tf
from keras.applications.mobilenet import MobileNet
from keras import backend as K
from flask import Flask
from flask import request
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
app = Flask(__name__)

# Load MobileNets
mobile_net = MobileNet()

# The following line is necessary because Flask runs on a different thread than the loaded model
graph = tf.get_default_graph()

# Load the deep convolutional recurrent extra crispy one neuron network
linear_regression = LinearRegression()
linear_regression.fit([[1], [2]], [[2], [4]])


@app.route('/hello')
def hello_world():
    return 'Hello, World!'


@app.route('/predict_regression', methods=['POST'])
def predict_regression():
    # Get input value
    X = int(request.form['X'])

    # Place input value in a list of lists
    X = [[X]]

    # Predict
    prediction = linear_regression.predict(X)

    # Return prediction
    return str(prediction)


@app.route('/predict_mobilenet', methods=['POST'])
def predict_mobilenet():
    # Get file descriptor
    file = request.files['input']

    # Read bytestream from file
    string = file.read()

    # Turn bytestream into 1D array
    data = np.fromstring(string, dtype='uint8')

    # Turn 1D array into 2D image
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    # Resize image so it fits model input
    image = cv2.resize(image, (224, 224))

    # Reshape image so it fits model input
    image = np.reshape(image, (1, *image.shape))

    # The following line is necessary because Flask runs on a different thread than the loaded model
    with graph.as_default():
        # Predict
        prediction = mobile_net.predict(image)

    # Return prediction
    return str(prediction)


@app.route('/predict_mobilenet_slow', methods=['POST'])
def predict_mobilenet_slow():
    K.clear_session()
    mobile_net = MobileNet()
    graph = tf.get_default_graph()

    # Get file descriptor
    file = request.files['input']

    # Read bytestream from file
    string = file.read()

    # Turn bytestream into 1D array
    data = np.fromstring(string, dtype='uint8')

    # Turn 1D array into 2D image
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    # Resize image so it fits model input
    image = cv2.resize(image, (224, 224))

    # Reshape image so it fits model input
    image = np.reshape(image, (1, *image.shape))

    # The following line is necessary because Flask runs on a different thread than the loaded model
    with graph.as_default():
        # Predict
        prediction = mobile_net.predict(image)

    # Return prediction
    return str(prediction)


if __name__ == '__main__':
    app.run()
