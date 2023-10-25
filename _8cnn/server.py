import numpy as np
import json
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import os

cwd = os.getcwd()
# print(cwd+"/cnn")
model = load_model(cwd+'/cnn/mnist_model.h5')
print('Model loaded: ', model.get_config()[
      "layers"][0]["config"]["batch_input_shape"])

app = Flask(__name__)


@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello World!'})


@app.route('/api/mnist', methods=['POST'])
def mnist():
    req_data = request.get_json(force=True)
    image_data = req_data['img']
    image = np.array(image_data).reshape(1, 28, 28, 1)
    print(image.shape)
    pred = model.predict(image)
    digit = np.argmax(pred)
    return jsonify({"digit": str(digit)})


if __name__ == '__main__':
    app.run(port=8000, debug=True)
