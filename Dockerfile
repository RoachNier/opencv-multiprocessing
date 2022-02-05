FROM python

WORKDIR /app

COPY . .

RUN pip install opencv-python
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install numpy

CMD ["python", "main.py"]
