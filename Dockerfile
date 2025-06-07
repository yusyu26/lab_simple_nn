FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libglib2.0-0 \
        libgl1       \
        libsm6       \
        libxrender1  \
        libxext6     \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

#Install Pytorch
RUN pip install --no-cache-dir \
        numpy \
        matplotlib \
        torch \
        torchvision

COPY . /app

CMD ["bash"]