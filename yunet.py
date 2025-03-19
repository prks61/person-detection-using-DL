from itertools import product

import numpy as np
import cv2 as cv

class YuNet:
    def __init__(self, modelPath, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3, topK=5000, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize) # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        faces = self._model.detect(image)
        return np.empty(shape=(0, 5)) if faces[1] is None else faces[1]


# import os
# from embed_image import YuNetArgumentParser, visualize, backend_target_pairs

# args = YuNetArgumentParser(input='C:/Users/PRINCE VERMA/Desktop/BTP/backend/person_detection/detection_app_api/Face_detect/pr.png', save=True)
# # backend_id = backend_target_pairs[args.backend_target][0]
# target_id = backend_target_pairs[args.backend_target][1]
# # print(backend_id)
# model_path = os.path.join(os.getcwd(), "detection_app_api", "Face_detect", "face_detection_yunet_2023mar_int8.onnx")
# model = YuNet(modelPath='C:/Users/PRINCE VERMA/Desktop/BTP/backend/person_detection/detection_app_api/Face_detect/face_detection_yunet_2023mar_int8.onnx',
#                 inputSize=[900, 900],
#                 confThreshold=args.conf_threshold,
#                 nmsThreshold=args.nms_threshold,
#                 topK=args.top_k,
#                 backendId=0,
#                 targetId=target_id)