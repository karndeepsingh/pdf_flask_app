#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import flask
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from flask import Flask, render_template, request, redirect, url_for, abort,flash
from PIL import Image
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
import os
from statistics import mode

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = [".pdf"]
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
model = None





def model():
    global model
    model = load_model("bestmodel.h5")

class_labels = ["tdp", "tds","unknown"]
def prediction(image,target_size = (224,224,3)):   
    image= image.resize(target_size)
    # Convert the image pixels to a numpy array
    image = img_to_array(image)
    image =image/255.0
    # Reshape data for the model
    image = np.expand_dims(image, axis=0)
    # Pass image into model to get encoded features
    preds= model.predict(image, verbose=0)

    pred_proba =preds[0][np.argmax(preds)]
    class_l = class_labels[np.argmax(preds)]

    return class_l,pred_proba

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("File"):
            uploaded_file = flask.request.files["File"]
            filename = secure_filename(uploaded_file.filename)
                
            if filename !="":
                file_ext = os.path.splitext(filename)[1]
                if file_ext not in app.config["UPLOAD_EXTENSIONS"]:
                    flash("Upload pdf file format")
                    abort(400)
                    
                else:    
                    pdf_file = flask.request.files["File"]
                    pdf_file.save(filename)
                    for pdf_file in os.listdir("."):
                            if pdf_file.endswith(".pdf"):
                                pages = convert_from_path(pdf_file,400)
                                counter = 1
                                for page in pages:
                                    myfile ='output'+"_"+ str(counter) +'.jpg'
                                    counter = counter + 1
                                    page.save(myfile, "JPEG")
                                    
                            images_path = [image_path for image_path in os.listdir(".") if image_path.endswith(".jpg")]
                           
                            for image_path in images_path:               
                                    image= Image.open(image_path)
                                    # preprocess the image and prepare it for classification
                                    class_l, pred_proba = prediction(image, target_size = (224,224))
                                  
                                    
                                  
                                    

                                    data["predictions"] = []
                                    # loop over the results and add them to the list of
                                    # returned predictions
                                    result= {"label": class_l, "probability": float(pred_proba)}
                                    data["predictions"].append(result)
                                    #delete the image from the directory
                                    os.remove(image_path)
                                    




                            # indicate that the request was a success
                            data["success"] = True
                    os.remove(filename) 



            


        # return the data dictionary as a JSON response
        return flask.jsonify(data)
        if  "File" not in flask.request.files:
            return flash("No file Present")

# start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    model()
    app.run(debug=True)








