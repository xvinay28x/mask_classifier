import tensorflow as tf
import os
from flask import Flask, render_template, request

model = tf.keras.models.load_model('Mask_Classifier.h5')

app = Flask(__name__)

@app.route('/')
def main():
    return render_template("index.html",img_ads = "static/image/send.png" )


@app.route('/predict', methods=['POST'])
def home():
    image = request.files["image"]
    save = image.save("static\image.jpg")
    load_image = tf.keras.preprocessing.image.load_img("static\image.jpg",target_size=(200,200))
    image_array = tf.keras.preprocessing.image.img_to_array(load_image)
    reshape_array = image_array.reshape(1,200,200,3)
    image = reshape_array/255
    result = model.predict(image)

    if result >= 0.5:
        result = "No Mask"
    else:
        result = "Mask"
    
    return render_template("index.html", result = result , img_ads = "static\image.jpg" )

if __name__ == "__main__": 
    app.run(debug=True)    