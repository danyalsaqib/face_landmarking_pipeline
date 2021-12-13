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

# Only function to be called is postprocess_image

def postprocess_image_embed(result):
    #find_pose(points)
    #name = name.replace('_', ' ')
    diction = {'embeddings': result.astype(int).tolist()}
    #diction['names'].append(name)
    #diction['embeddings'].append(result[0].astype(int).tolist())
    #diction['embeddings']
    with open('dictionary.json', 'w') as handle:
        json.dump(diction, handle)
    return result

