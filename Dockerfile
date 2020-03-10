FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends  software-properties-common \
    nginx \
    wget && \
    add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3.8 \
    python3.8-dev \
    python3-pip

# update pip
RUN python3.8 -m pip install pip --upgrade
RUN python3.8 -m pip install wheel

# install the requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY src /opt/src

WORKDIR /opt/src/main