#detector

from keras.models import *
from keras.layers import *
import tensorflow as tf

class L2Norm(Layer):
    '''
    Code borrows from https://github.com/flyyufelix/cnn_finetune
    '''
    def __init__(self, weights=None, axis=-1, gamma_init='zero', n_channels=256, scale=10, **kwargs):
        self.axis = axis
        self.gamma_init = tf.keras.initializers.get(gamma_init)
        self.initial_weights = weights
        self.n_channels = n_channels
        self.scale = scale
        super(L2Norm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.gamma = K.variable(self.gamma_init((self.n_channels,)), name='{}_gamma'.format(self.name))
        self.trainable_weights1 = [self.gamma]
        self.built = True

    def call(self, x, mask=None):
        norm = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)) + K.epsilon()
        x = x / norm * self.gamma
        return x

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super(L2Norm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def s3fd_keras():
    inp = Input((None,None,3))

    conv1_1 = Conv2D(filters=64, kernel_size=3, padding="same", name="conv1_1", activation="relu")(inp)
    conv1_2 = Conv2D(filters=64, kernel_size=3, padding="same", name="conv1_2", activation="relu")(conv1_1)
    maxpool1 = MaxPooling2D()(conv1_2)

    conv2_1 = Conv2D(filters=128, kernel_size=3, padding="same", name="conv2_1", activation="relu")(maxpool1)
    conv2_2 = Conv2D(filters=128, kernel_size=3, padding="same", name="conv2_2", activation="relu")(conv2_1)
    maxpool2 = MaxPooling2D()(conv2_2)

    conv3_1 = Conv2D(filters=256, kernel_size=3, padding="same", name="conv3_1", activation="relu")(maxpool2)
    conv3_2 = Conv2D(filters=256, kernel_size=3, padding="same", name="conv3_2", activation="relu")(conv3_1)
    conv3_3 = Conv2D(filters=256, kernel_size=3, padding="same", name="conv3_3", activation="relu")(conv3_2)
    f3_3 = conv3_3
    maxpool3 = MaxPooling2D()(conv3_3)

    conv4_1 = Conv2D(filters=512, kernel_size=3, padding="same", name="conv4_1", activation="relu")(maxpool3)
    conv4_2 = Conv2D(filters=512, kernel_size=3, padding="same", name="conv4_2", activation="relu")(conv4_1)
    conv4_3 = Conv2D(filters=512, kernel_size=3, padding="same", name="conv4_3", activation="relu")(conv4_2)
    f4_3 = conv4_3
    maxpool4 = MaxPooling2D()(conv4_3)

    conv5_1 = Conv2D(filters=512, kernel_size=3, padding="same", name="conv5_1", activation="relu")(maxpool4)
    conv5_2 = Conv2D(filters=512, kernel_size=3, padding="same", name="conv5_2", activation="relu")(conv5_1)
    conv5_3 = Conv2D(filters=512, kernel_size=3, padding="same", name="conv5_3", activation="relu")(conv5_2)
    f5_3 = conv5_3
    maxpool5 = MaxPooling2D()(conv5_3)


    # ========== Note ==========
    # Be careful about the zeropadding difference when strides >= 2
    fc6 = ZeroPadding2D(3)(maxpool5)
    fc6 = Conv2D(filters=1024, kernel_size=3, name="fc6", activation="relu")(fc6)
    fc7 = Conv2D(filters=1024, kernel_size=1, name="fc7", activation="relu")(fc6)
    ffc7 = fc7
    conv6_1 = Conv2D(filters=256, kernel_size=1, name="conv6_1", activation="relu")(fc7)
    f6_1 = conv6_1
    conv6_2 = ZeroPadding2D()(conv6_1)
    conv6_2 = Conv2D(filters=512, kernel_size=3, strides=2, name="conv6_2", activation="relu")(conv6_2)
    f6_2 = conv6_2
    conv7_1 = Conv2D(filters=128, kernel_size=1, name="conv7_1", activation="relu")(f6_2)
    f7_1 = conv7_1
    conv7_2 = ZeroPadding2D()(conv7_1)
    conv7_2 = Conv2D(filters=256, kernel_size=3, strides=2, name="conv7_2", activation="relu")(conv7_2)
    f7_2 = conv7_2

    f3_3 = L2Norm(n_channels=256, scale=10, name="conv3_3_norm")(f3_3)
    f4_3 = L2Norm(n_channels=512, scale=8, name="conv4_3_norm")(f4_3)
    f5_3 = L2Norm(n_channels=512, scale=5, name="conv5_3_norm")(f5_3)

    cls1 = Conv2D(filters=4, kernel_size=3, padding="same", name="conv3_3_norm_mbox_conf")(f3_3)
    reg1 = Conv2D(filters=4, kernel_size=3, padding="same", name="conv3_3_norm_mbox_loc")(f3_3)
    cls2 = Conv2D(filters=2, kernel_size=3, padding="same", name="conv4_3_norm_mbox_conf")(f4_3)
    reg2 = Conv2D(filters=4, kernel_size=3, padding="same", name="conv4_3_norm_mbox_loc")(f4_3)
    cls3 = Conv2D(filters=2, kernel_size=3, padding="same", name="conv5_3_norm_mbox_conf")(f5_3)
    reg3 = Conv2D(filters=4, kernel_size=3, padding="same", name="conv5_3_norm_mbox_loc")(f5_3)
    cls4 = Conv2D(filters=2, kernel_size=3, padding="same", name="fc7_mbox_conf")(ffc7)
    reg4 = Conv2D(filters=4, kernel_size=3, padding="same", name="fc7_mbox_loc")(ffc7)

    cls5 = Conv2D(filters=2, kernel_size=3, padding="same", name="conv6_2_mbox_conf")(f6_2)
    reg5 = Conv2D(filters=4, kernel_size=3, padding="same", name="conv6_2_mbox_loc")(f6_2)
    cls6 = Conv2D(filters=2, kernel_size=3, padding="same", name="conv7_2_mbox_conf")(f7_2)
    reg6 = Conv2D(filters=4, kernel_size=3, padding="same", name="conv7_2_mbox_loc")(f7_2)

    def get_chunk(x, c):
        return tf.split(x, c, axis=-1)
    chunk = Lambda(lambda x: get_chunk(x, 4))(cls1)
    bmax = Lambda(lambda chunk: K.maximum(K.maximum(chunk[0], chunk[1]), chunk[2]))(chunk)
    cls1 = Concatenate(axis=-1)([bmax, chunk[3]])

    return Model(inp, [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6])


import numpy as np
from scipy import special
#from .model import s3fd_keras

class S3FD():
    def __init__(self, weights_path="detector_weights/s3fd_keras_weights.h5"):
        self.net = s3fd_keras()
        self.net.load_weights(weights_path)

    def detect_face(self, image):
        bboxlist = self.detect(self.net, image)
        keep = self.nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]
        return bboxlist

    def detect(self, net, img):
        def softmax(x, axis=-1):
            return np.exp(x - special.logsumexp(x, axis=axis, keepdims=True))
        img = img - np.array([104, 117, 123])
        if img.ndim == 3:
            img = img[np.newaxis, ...]
        elif img.ndim == 5:
            img = np.squeeze(img)

        BB, HH, WW, CC = img.shape
        olist = net.predict(img) # output a list of 12 predicitons in different resolution

        bboxlist = []
        for i in range(len(olist) // 2):
            olist[i * 2] = softmax(olist[i * 2], axis=-1)
        olist = [oelem for oelem in olist]
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            FB, FH, FW, FC = ocls.shape  # feature map size
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            anchor = stride * 4
            poss = zip(*np.where(ocls[:, :, :, 1] > 0.05))
            for Iindex, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[0:1, hindex, windex, 1]
                loc = oreg[0:1, hindex, windex, :]
                priors = np.array([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = self.decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                bboxlist.append([x1, y1, x2, y2, score])
        bboxlist = np.array(bboxlist)
        if 0 == len(bboxlist):
            bboxlist = np.zeros((1, 5))
        return bboxlist

    @staticmethod
    def decode(loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """
        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    @staticmethod
    def nms(dets, thresh):
        if 0 == len(dets):
            return []
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
            w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
            ovr = w * h / (areas[i] + areas[order[1:]] - w * h)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep



import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from keras import backend as K

#from .s3fd.s3fd_detector import S3FD
#from .mtcnn.mtcnn_detector import MTCNN
#from .landmarks_detector import FANLandmarksDetector

#FILE_PATH = Path(__file__).parent.resolve()

class BaseFaceDetector():
    def __init__(self):
        pass

    def detect_face(self):
        raise NotImplementedError

'''class MTCNNFaceDetector(BaseFaceDetector):
    def __init__(self, weights_path=FILE_PATH / "mtcnn"):
        self.face_detector = MTCNN(weights_path)

    def detect_face(self, image):
        # Output bbox coordinate has ordering (y0, x0, y1, x1)
        return self.face_detector.detect_face(image)

    def batch_detect_face(self, image):
        raise NotImplementedError'''

class S3FaceDetector(BaseFaceDetector):
    def __init__(self, weights_path="detector_weights/s3fd_keras_weights.h5"):
        self.face_detector = S3FD(weights_path)

    def detect_face(self, image):
        # Output bbox coordinate has ordering (y0, x0, y1, x1)
        return self.face_detector.detect_face(image)

    def batch_detect_face(self, image):
        raise NotImplementedError



class FaceAlignmentDetector(BaseFaceDetector):
    def __init__(self,
                 fd_weights_path="detector_weights/s3fd_keras_weights.h5",
                 lmd_weights_path="detector/detector_weights/2DFAN-4_keras.h5",
                 fd_type="s3fd"):
        self.fd_type = fd_type.lower()
        if fd_type.lower() == "s3fd":
            self.fd = S3FaceDetector(fd_weights_path)
        elif fd_type.lower() == "mtcnn":
            self.fd = MTCNNFaceDetector()
        else:
            raise ValueError(f"Unknown face detector {face_detector}.")

        self.lmd_weights_path = lmd_weights_path
        self.lmd = None

    def build_FAN(self):
        self.lmd = FANLandmarksDetector(self.lmd_weights_path)

    def preprocess_s3fd_bbox(self,bbox_list):
        # Convert coord (y0, x0, y1, x1) to (x0, y0, x1, y1)
        return [np.array([bbox[1], bbox[0], bbox[3], bbox[2], bbox[4]]) for bbox in bbox_list]

    def detect_face(self, image, with_landmarks=True):
        """
        Returns:
            bbox_list: bboxes in [x0, y0, x1, y1] ordering (x is the vertical axis, y the height).
            landmarks_list: landmark points having shape (68, 2) with ordering [[x0, y0], [x1, y1], ..., [x67, y67].
        """
        if self.fd_type == "s3fd":
            bbox_list = self.fd.detect_face(image)
        elif self.fd_type == "mtcnn":
            bbox_list = self.fd.detect_face(image)
        if len(bbox_list) == 0:
            return [], []

        if with_landmarks:
            if self.lmd == None:
                print("Building FAN for landmarks detection...")
                self.build_FAN()
                print("Done.")
            landmarks_list = []
            for bbox in bbox_list:
                pnts = self.lmd.detect_landmarks(image, bounding_box=bbox)[-1]
                landmarks_list.append(np.array(pnts))
            landmarks_list = [self.post_process_landmarks(landmarks) for landmarks in landmarks_list]
            bbox_list = self.preprocess_s3fd_bbox(bbox_list)
            return bbox_list, landmarks_list
        else:
            bbox_list = self.preprocess_s3fd_bbox(bbox_list)
            return bbox_list

    def batch_detect_face(self, images, **kwargs):
        raise NotImplementedError


#for getting bboxes we have fd
