
# !/usr/bin/env python
# coding: utf-8

# In[4]:


import uuid
import numpy as np
import datetime
import flask
import json
import base64
import time
import io
import os
import random
from threading import Thread
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, render_template, request, redirect, url_for, abort, flash, jsonify, make_response
from PIL import Image
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
from statistics import mode
from functools import wraps
from multiprocessing import Process, Pool, cpu_count
from azure.cosmosdb.table.tableservice import TableService
from flask_jwt_extended import JWTManager
from flask_jwt_extended import (create_access_token, 
                                create_refresh_token,
                                jwt_required,
                                jwt_refresh_token_required,
                                get_jwt_identity,
                                get_raw_jwt)

import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = None

# initialize our Flask application and the Keras

app = flask.Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = [".pdf"]
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024
app.config['ENV'] = 'production'
app.config['DEBUG'] = True
app.config['TESTING'] = False

app.config['JWT_SECRET_KEY'] = 'kldLaDFI7584%38&mFyt&84cvT593(jfkdJkg@1'
jwt = JWTManager(app)

model = load_model("bestmodel.h5")
print('model is: ')
print(model)

class_labels = ["tdp", "tds", "unknown"]

username = "MachinelearningUser"
password = "caTchme!fUc@n"


def prediction(image, target_size=(224, 224, 3)):
    global model
    image = image.resize(target_size)
    # Convert the image pixels to a numpy array
    image = img_to_array(image)
    image = image / 255.0
    # Reshape data for the model
    image = np.expand_dims(image, axis=0)
    # Pass image into model to get encoded features
    preds = model.predict(image, verbose=0)

    pred_proba = preds[0][np.argmax(preds)]
    class_l = class_labels[np.argmax(preds)]

    return class_l, pred_proba


def make_prediction_for_image(file_name):
    image = Image.open(file_name)
    class_l, pred_proba = prediction(image, target_size=(224, 224))
    result = {"label": class_l, "probability": float(pred_proba)}
    return result


@app.route("/predict", methods=["POST"])
@jwt_required
def predict():
    start_time = datetime.datetime.utcnow()
    table_service = TableService(account_name='marketsharepoc2514228028',
                                 account_key='bwHjIl8l3QFN1kRtSN2CNqS3Cc891Zjtp7/KkNokJiYLCm4X416QATlEKkPkkQjKU5n730rqmCrNdRBRpSDqrQ==')

    data = {"success": False}
    request_data = json.loads(flask.request.data.decode('utf-8'))
    initial_encode_char = str(request_data.get('pdf')[:64])

    byte_data = base64.b64decode(request_data.get('pdf'), validate=True)

    if byte_data[0:4] != b'%PDF':
        raise ValueError('Missing the PDF file signature')

    dir_name = random.randint(10000, 100000)
    dir_path = f'{os.getcwd()}/{dir_name}'
    print(dir_path)
    os.mkdir(dir_path)

    file_path = f"{dir_path}/x.pdf"
    file_3 = open(file_path, "wb")
    file_3.write(byte_data)
    file_3.close()

    os.chdir(dir_path)

    pages = convert_from_path(file_path, 400)
    counter = 1
    data_prediction = list()
    for page in pages:
        page.save(f'{dir_path}/{counter}.jpg', "JPEG")
        result = make_prediction_for_image(f'{dir_path}/{counter}.jpg')
        data_prediction.append(result)
        os.remove(f'{dir_path}/{counter}.jpg')
        counter += 1

    os.remove(file_path)

    data["success"] = True
    data["prediction"] = data_prediction
    print(data)
    os.chdir('..')
    os.rmdir(dir_path)

    task = {
        'pdf_encoded': initial_encode_char,
        'success': data["success"],
        'time_taken': f'{(datetime.datetime.utcnow() - start_time).seconds} Sec',
        'prediction': f"""{data["prediction"]}""",
        'PartitionKey': datetime.datetime.utcnow().strftime('%m-%Y'),
        'RowKey': str(uuid.uuid4().hex)
    }
    table_service.insert_entity('unileverlogs', task)
    return flask.jsonify(data)


@app.route("/login", methods=["POST"])
def login():
    request_data = json.loads(flask.request.data.decode('utf-8'))
    response = {}
    status_code = 401

    u_name = request_data.get('username', '')
    u_password = request_data.get('password', '')
    if not u_name or not u_password:
        response = {'msg': 'Username & Password both are required'}
        status_code = 400

    elif u_name == username and u_password == password:
        access_token = create_access_token(identity=u_name,
                                           expires_delta=datetime.timedelta(days=36500))
        print(f'access_token is: {access_token}')
        response = {'token': access_token, 'msg': 'Login successful'}
        status_code = 200

    elif u_name != username or u_password != password:
        response = {'msg': 'Invalid credentials'}

    response = make_response(jsonify(response), status_code)
    response.headers["Content-Type"] = "application/json"
    return response


# start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run()
