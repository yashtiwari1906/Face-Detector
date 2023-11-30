from model import DetectorModel
from flask import Flask, render_template, request
from PIL import Image
import numpy as np 
import cv2 
import json

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        # if isinstance(obj):
        #     # Return a serializable representation of YourCustomClass
        return obj.to_dict()  # Replace with your custom method


app = Flask(__name__)

@app.route('/')
def home():
    return {"text": "hello world"}

@app.route('/predict', methods=['POST'])
def upload():
    image_list = request.get_json()["image_array"]
    arrImg = np.array(image_list)
    frame = cv2.cvtColor(np.float32(arrImg), cv2.COLOR_RGB2BGR)
    detector = DetectorModel("s3fd")
    output = detector.predict(frame) 

    return {"coordianates": output}

if __name__ == '__main__':
    app.run(debug=True, port=5001)
