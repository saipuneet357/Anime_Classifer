from flask import Flask, redirect, url_for, render_template, request
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/')
def home():
	return render_template("index.html")	

@app.route("/<usr>")
def user(usr):
	return "<h1>{usr}</h1>".format(usr=usr)
	
@app.route('/login', methods=["POST", "GET"])
def login():
	if request.method == "POST":
		image = request.files['k']
		img_str = image.read()
		nparr = np.fromstring(img_str, np.uint8)
		img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR
		while True:
			cv2.imshow('image', img_np)
			if cv2.waitKey(0) & 0b11111111 == ord('q'):
				break
		return redirect(url_for("home"))
	else:
		return render_template('login.html')


if __name__ == '__main__':
	app.run(debug=True)
