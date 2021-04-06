from flask import Flask, redirect, url_for, render_template, request
import numpy as np
import cv2
from load import predict
import os

app = Flask(__name__)

	
@app.route('/', methods=["POST", "GET"])
def home():
	if request.method == "POST":
		image = request.files['k']
		img_str = image.read()
		nparr = np.fromstring(img_str, np.uint8)
		img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR
		
		args = {'image':img_np, 'model':'model.h5', 'labels':'lb.pickle'}
		
		output, label = predict(**args)
		
		try:
			os.remove('static/images/image.jpg')
			print('old file removed')
		except:
			print('file not available')
		
		cv2.imwrite('static/images/image.jpg', output)
		
		
		return redirect(url_for("show", label=label))
	else:
		return render_template('index.html')


@app.route('/show/<label>')
def show(label):
	return render_template('show.html', type=label)


if __name__ == '__main__':
	app.run()
