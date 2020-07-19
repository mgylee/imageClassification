
import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow import keras

app = Flask(__name__)
saved_model = keras.models.load_model('model/model.h5')

from preprocess import preprocess
from inference import get_class_name

@app.route('/', methods=['GET', 'POST'])
def upload():
	if request.method == 'GET':
		return render_template('index.html')
	if request.method == 'POST':
		print(request.files)
		if 'file' not in request.files:
			print('file not uploaded')
			return
		f = request.files['file']
		image_name = f.filename
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'static', secure_filename(f.filename))
		f.save(file_path)

		image = preprocess(file_path)
		num_class = saved_model.predict_classes(image)
		class_name = get_class_name(num_class[0])
		return render_template('result.html', label = class_name, img = image_name)

if __name__ == '__main__':
	app.run(debug=True)