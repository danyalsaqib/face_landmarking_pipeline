import numpy as np
import cv2 as cv
import os
import onnx
import onnxruntime
from onnx import numpy_helper
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf
from retinaface.commons import preprocess, postprocess
import json
from os import path
from helper_functions import warp_and_crop_face, get_reference_facial_points

# Only function to be called is preprocess_image

def preprocess_image(crop_img, points):
    output_size=(224 , 224)
    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    facial5points = np.reshape(points, (2, 5))
    reference_5pts = get_reference_facial_points(output_size, inner_padding_factor, outer_padding, default_square)
    dst_img = warp_and_crop_face(crop_img, facial5points, reference_pts=reference_5pts, crop_size=output_size)
    mid_x, mid_y = int(112), int(112)
    cw2, ch2 = int(150/2), int(150/2)
    crop_img2 = dst_img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    #cv.imwrite('alignedImage.jpg', crop_img2)
    #Recognition
    img = cv.resize(crop_img2, dsize=(112, 112), interpolation=cv.INTER_AREA)
    img.resize((1, 3, 112, 112))
    data = json.dumps({'data': img.tolist()})
    a = np.array(json.loads(data)['data']).astype('float32')
    return a, data

