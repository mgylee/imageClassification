
import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow import keras

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert inputted image to an array
def process_image(file_path):
    image_array = []
    image_prep = keras.preprocessing.image.load_img(file_path, target_size = (28, 28, 3))
    image_prep = keras.preprocessing.image.img_to_array(image_prep)
    image_prep = image_prep/255
    image_array.append(image_prep)
    return np.array(image_array)

# Load model and make prediction
def get_class_prediction(image_array):
    classes = {
        0 : 'buildings',
        1 : 'forest',
        2 : 'glacier',
        3 : 'mountain',
        4 : 'sea',
        5 : 'street'
    }
    saved_model = keras.models.load_model('model/model.h5')
    class_index = saved_model.predict_classes(image_array)
    return classes[class_index[0]]

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        f = request.files['file']
        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if f and allowed_file(f.filename):
            image_name = f.filename
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'static', secure_filename(f.filename))
            f.save(file_path)

            image = process_image(file_path)
            class_name = get_class_prediction(image).capitalize()
            return render_template('upload.html', label = class_name, img = image_name)
    return

if __name__ == '__main__':
    app.run(debug=True)