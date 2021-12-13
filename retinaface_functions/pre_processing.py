import cv2 as cv
import numpy as np
import pandas as pd
from os import path
import os
from inference import infer_image_landmark
from post_processing import postprocess_image_landmark
from embedding_functions.pre_processing import preprocess_image_embed
from embedding_functions.inference import infer_image_embed
from embedding_functions.post_processing import postprocess_image_embed

# Only function to be called is preprocess_image

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    print("a: ", a.shape)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def resize_image(img, scales, allow_upscaling):
    img_h, img_w = img.shape[0:2]
    target_size = scales[0]
    max_size = scales[1]

    if img_w > img_h:
        im_size_min, im_size_max = img_h, img_w
    else:
        im_size_min, im_size_max = img_w, img_h

    im_scale = target_size / float(im_size_min)
    if not allow_upscaling:
        im_scale = min(1.0, im_scale)

    if np.round(im_scale * im_size_max) > max_size:
        im_scale = max_size / float(im_size_max)

    if im_scale != 1.0:
        img = cv.resize(
            img,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv.INTER_LINEAR
        )

    return img, im_scale

def ppc_retina(img, allow_upscaling):
    pixel_means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pixel_stds = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    pixel_scale = float(1.0)
    scales = [1024, 1980]

    img, im_scale = resize_image(img, scales, allow_upscaling)
    img = img.astype(np.float32)
    im_tensor = np.zeros((1, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

    # Make image scaling + BGR2RGB conversion + transpose (N,H,W,C) to (N,C,H,W)
    for i in range(3):
        im_tensor[0, :, :, i] = (img[:, :, 2 - i] / pixel_scale - pixel_means[2 - i]) / pixel_stds[2 - i]

    return im_tensor, img.shape[0:2], im_scale

# This function currently takes an image_path, and returns cv2 object
def preprocess_image_landmark(image_path):
    print("\n*********************")
    print("Preprocessing for Face Detection and Landmarking")
    file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
    output_size=(224 , 224)
    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    threshold = 0.5
    #diction = {'names':[], 'embeddings':[]}
    ## Detection with mtcnn
    if path.exists(image_path) == False:
        print("Invalid File Path")
        return -1
    else:
        image = cv.imread(image_path)
        print(str(type(image))[8:-2])
        if (str(type(image))[8:-2] == 'NoneType'):
            print("Invalid Image")
            return -1
        print("image shape: ", image.shape)
        #pixels = plt.imread(image_path)
        im_tensor, im_info, im_scale = ppc_retina(image, allow_upscaling=True)
        print("Image Tensor Shape: ", im_tensor.shape)
        return [im_tensor, im_info, im_scale], image
        
def get_representation(file_path):
    lol = preprocess_image_landmark(file_path)
    if lol != -1:
        lol = infer_image_landmark(lol)
        lol = postprocess_image_landmark(lol)
        if lol != -1:
            lol = preprocess_image_embed(lol)
            lol = infer_image_embed(lol)
            lol = postprocess_image_embed(lol)
            print("Final Output: ", lol.shape)
    return lol
                
if __name__ == '__main__':
    im = get_representation('images/Aamir Hussain Liaquat_img_1.jpg')
    
