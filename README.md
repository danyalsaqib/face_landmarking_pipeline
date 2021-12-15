# face_landmarking_pipeline
The link to ONNX models used can be found at: https://drive.google.com/drive/folders/1N2WusFEbFUaaj82NxfjADcRT_VWAG_Uo?usp=sharing

These models should be placed in 'face_landmarking_pipeline/retinaface_functions/'. The preprocessing file 'face_landmarking_pipeline/retinaface_functions/pre_processing.py' contains a main function that can be used to run the entire pipeline.

## Dockerfile
The dockerfile contains instructions to first install necessary libraries, and then copy the 'retinaface_functions/' directory into a local docker directory:
```
COPY retinaface_functions /usr/local/retinaface_functions
```
The docker uses the tensorflow container as its parent:
```
FROM nvcr.io/nvidia/tensorflow:21.11-tf2-py3
```
To build the docker, simply use the following command:
```
docker build -t face_pipeline:retina_1 .
```
To run the docker after successfully building it, use the following command:
```
docker run face_pipeline:retina_1
```

## Running the complete pipeline
The 'retinaface_functions/pre_processing.py' file has a main functon, that can run the entire pipeline with 1 call. Its usage is as follows:
```
python3 pre_processing.py --path 'path/to/image_file.jpg'
```

## Individual Functions

### RetinaFace Functions
1. pre_processing.py:

Preprocesses image, and outputs an image tensor. Only call 'preprocess_image' function

2. inference.py:

Performs inference on image tensor, and outputs RetinaFace output. Only call 'infer_image' function

2. post_processing.py:

Postprocesses RetinaFace output, and outputs cropped image and 'relative landmarks'. Only call 'postprocess_image' function

### Embedding Functions
Should be given output of postprocessing of RetinaFace
1. pre_processing.py:

Preprocesses previous results, and outputs aligned images data. Only call 'preprocess_image_embed' function

2. inference.py:

Performs inference on aligned images, and outputs Raw Embeddings. Only call 'infer_image' function

2. post_processing_embed.py:

Postprocesses Raw Embeddings, and outputs Final Embeddings. Only call 'postprocess_image_embed' function
