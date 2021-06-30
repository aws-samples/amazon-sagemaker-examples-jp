FROM python:3.8.10-slim-buster

RUN apt-get update; \
    apt-get install -y cmake traceroute\
    ; \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install awsiotsdk numpy tensorflow-cpu==2.4.1 Pillow
WORKDIR /app

COPY ./IoTPublisher.py /app/IoTPublisher.py
COPY ./gan.h5 /app/gan.h5
COPY ./classifier.h5 /app/classifier.h5

CMD ["python3", "/app/IoTPublisher.py"]
