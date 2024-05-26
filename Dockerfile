FROM tensorflow/tensorflow:2.12.0-gpu

RUN pip install tensorrt matplotlib pillow pydantic tqdm

WORKDIR /app

COPY tfrecords tfrecords 

COPY horizon horizon
 
COPY train.py train.py

CMD python train.py