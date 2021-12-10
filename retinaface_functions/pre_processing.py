import cv2 as cv
import numpy as np
from os import path

# Only function to be called is preprocess_image

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
def preprocess_image(image_path):
    file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
    output_size=(224 , 224)
    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    threshold = 0.5
    #diction = {'names':[], 'embeddings':[]}
    ## Detection with mtcnn
    try:
        if path.exists(image_path) ==False:
            raise PathNotFound
        else:
            image = cv.imread(image_path)
            print("image shape: ", image.shape)
            #pixels = plt.imread(image_path)
            im_tensor, _, _ = ppc_retina(image, allow_upscaling=True)
            return im_tensor