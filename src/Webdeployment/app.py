from distutils.log import debug
import os
from flask import Flask, render_template, request
from src.models.predict_model import *
import os
import cv2
import tensorflow as tf
import base64
import json

def image2numpy(data):
    image = tf.keras.preprocessing.image.load_img(data,target_size=(28,28),grayscale=True,interpolation='nearest')
    

    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    # with open("file_orig.npy" , 'wb')as f:
    #     np.save(f, input_arr)
    # input_arr = np.array([input_arr])  # Convert single image to a batch.
    # cv2.imwrite("1_post.png", gray_image)
    # cv2.imwrite("1_post2.png", res)
    return input_arr;



app = Flask(__name__,static_url_path='', 
            static_folder='static')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit_draw", methods = ['GET','POST'])
def get_output_draw():
    if request.method == 'POST':
        img = request.form['data'];
        img_pathForFileRead =  "./src/Webdeployment/static/predImage1.png" 
        new_data = img.replace('data:image/png;base64,', '')
        imgdata = base64.b64decode(new_data)
        with open(img_pathForFileRead, 'wb') as f:
            f.write(imgdata)
        imgArray = image2numpy(img_pathForFileRead)
        resultNum = predict_model(1,imgArray)
        print("Result : " , resultNum)
    
    return render_template("index.html", prediction = resultNum, img_path = "predImage1.png" )


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        filename, file_extension = os.path.splitext(img.filename)
        img_pathForFileRead =  "./src/Webdeployment/static/" + "predImage" + file_extension
        img.save(img_pathForFileRead)
        imgData = image2numpy(img_pathForFileRead)
        number = predict_model(1,imgData)
    return render_template("index.html", prediction = number, img_path = f"predImage{file_extension}" )






if __name__ == "__main__":
    app.run(debug=True)
    