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
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    img = Image.open(file).convert('RGB')
    arrImg = np.array(img)

    frame = cv2.cvtColor(arrImg, cv2.COLOR_RGB2BGR)
    detector = DetectorModel("s3fd")
    output = detector.predict(frame) 
    if file.filename == '':
        return 'No selected file'

    if file:
        return {"mode": img.mode, "size": img.size, "coordianates": output}
if __name__ == '__main__':
    app.run(debug=True, port=5001)
