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
    pip install onnxruntime && \
    pip install -U tf2onn

RUN git clone https://github.com/danyalsaqib/face_landmarking_pipeline && \
    cd face_landmarking_pipeline/retinaface_functions && \
    gdown --id 16aJ_uiDWeggv0V7i9VBSzrdIOUhK84Qr && \
    gdown --id 15vfOurDqZDsZfxWoGrY2eIWeE6Uov5Zt

CMD python3 pre_processing.py
