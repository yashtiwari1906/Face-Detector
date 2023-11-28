from s3fd import FaceAlignmentDetector


class DetectorModel: 
    def __init__(self, model) -> None:
        self.model = model 
        self.face_coordinates = [] 
        self.load() 

    def load(self):
        self.detector_model = FaceAlignmentDetector(lmd_weights_path="weights/s3fd_keras_weights.h5")
    
    def preprocess(self, frame):
        pass 

    def predict(self, frame):
        bboxes = self.detector_model.detect_face(frame, with_landmarks = False)
        for idx, item in enumerate(bboxes):
            p1 = (int(item[1]), int(item[0]))
            p2 = (int(item[3]), int(item[2]))
            self.face_coordinates.append((p1, p2))

        return self.face_coordinates    

    def postprocess(self, frame):
        pass 

