FROM nvcr.io/nvidia/tensorflow:21.11-tf2-py3

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    apt install git-all && \
    pip install gdown && \
    pip install opencv-python && \
    pip install matplotlib && \
    python -m pip install -U scikit-image && \
    pip install retina-face && \
    pip install mtcnn && \
    pip install numpy protobuf==3.16.0 && \
    pip install onnx && \
    pip install onnxruntime \

COPY retinaface_functions /usr/local/retinaface_functions
