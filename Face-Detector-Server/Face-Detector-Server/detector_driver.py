import json
from typing import Dict, Union
import warnings
from .s3fd import FaceAlignmentDetector
import gdown
import kserve
import logging
import numpy as np 
import cv2 
from kserve.utils.utils import get_predict_input, get_predict_response
from kserve.errors import InferenceError, ModelMissingError
from kserve import Model, ModelServer, model_server, InferRequest, InferOutput, InferResponse
warnings.filterwarnings('ignore')

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class DetectorModel(kserve.Model): 
    def __init__(self,  name: str, model_dir: str) -> None:
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.face_coordinates = [] 
        self.load() 

    def download_weights(self):
        print("downloading weights....")
        try:
            url = "https://drive.google.com/drive/folders/13wfCyRK6DlV3omYSdI-N1sN5NGb7WN5p?usp=drive_link"
            gdown.download_folder(url, quiet=True, use_cookies=False)
            print("weights downloaded successfully")
        except Exception as e:
            raise RuntimeError("some error occured while downloading weights.", e)

    def load(self):
        self.download_weights()
        self.detector_model = FaceAlignmentDetector(lmd_weights_path="detector_weights/s3fd_keras_weights.h5")
        self.ready = True

    def preprocess(self, payload, headers):
        
        inputs = get_predict_input(payload)
        return inputs, headers
        
    def predict(self, payload, headers)-> Union[Dict, InferResponse]:
        try:
            image_list = payload[0]
            arrImg = np.array(image_list)
            frame = cv2.cvtColor(np.float32(arrImg), cv2.COLOR_RGB2BGR)
            bboxes = self.detector_model.detect_face(frame, with_landmarks = False)
            face_coordinates = [] 
            for idx, item in enumerate(bboxes):
                p1 = (int(item[1]), int(item[0]))
                p2 = (int(item[3]), int(item[2]))
                face_coordinates.append((p1, p2))

            infer_output = InferOutput(name="output-0", shape=list(np.array(face_coordinates).shape), datatype="FP32", data=face_coordinates)

            return infer_output 

        except Exception as e:
            raise InferenceError(str(e))

    def postprocess(self, payload, headers):
        return InferResponse(model_name=self.name, infer_outputs=[payload], response_id=123)
         
        

