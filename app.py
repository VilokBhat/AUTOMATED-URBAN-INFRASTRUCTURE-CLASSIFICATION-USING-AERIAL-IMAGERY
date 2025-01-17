from flask import Flask, render_template, request
from keras.preprocessing import image
#import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from keras.models import load_model
#import tensorflow as tf

# from PIL import Image, ImageEnhance
# import numpy as np
# import random



app = Flask(__name__)

model = load_model("model.h5")

def predict_label(img_path):
    img = load_img(img_path, target_size=(200, 200))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    predictions = np.argmax(model.predict(img))

    return predictions




@app.route('/train_reslt', methods=['GET', 'POST'])
def train_reslt():
	return render_template('results.html')

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		class_ = predict_label(img_path)
#['Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Pond', 'Port', 'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
		dataset_labels = {
			'Airport':0, 
			'BareLand':1, 
			'BaseballField':2, 
			'Beach':3, 
			'Bridge':4, 
			'Center':5, 
			'Church':6, 
			'Commercial':7, 
			'DenseResidential:':8, 
			'Desert':9, 
			'Farmland':10, 
			'Forest':11, 
			'Industrial':12, 
			'Meadow':13, 
			'MediumResidential':14, 
			'Mountain':15, 
			'Park':16, 
			'Parking':17, 
			'Playground':18, 
			'Pond':19, 
			'Port':20, 
			'RailwayStation':21, 
			'Resort':22, 
			'River':23, 
			'School':24, 
			'SparseResidential':25, 
			'Square':26, 
			'Stadium':27, 
			'StorageTanks':28, 
			'Viaduct':29
					}

		def getlabel(n):
			for x , y in dataset_labels.items(): 
				if n==y:
					return x

		cls_name = getlabel(class_)

	return render_template("index.html", prediction = cls_name, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
