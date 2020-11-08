FROM tensorflow/tensorflow:1.11.0-devel-gpu-py3

MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

USER root

# install basic tools, nodejs 10.x and yarn
RUN apt-get update
RUN apt -y install make gettext unzip zip nano diffutils git \
	&& pip3 install --upgrade pip \
	&& pip3 install --use-feature=2020-resolver awscli \
	&& apt clean all \
	&& rm -fr /root/.cache

RUN mkdir -p /usr/local/nvidia/lib64/
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/nvidia/lib64/libcuda.so.1

# add user genie-toolkit
RUN useradd -ms /bin/bash -r genie-toolkit

ARG GENIE_VERSION=master
RUN git clone https://github.com/stanford-oval/genie-k8s /opt/genie-toolkit/
WORKDIR /opt/genie-toolkit/
RUN git checkout ${GENIE_VERSION}

ARG MaSP_VERSION=master
RUN git clone https://github.com/nayamamura/MaSP /opt/MaSP/
WORKDIR /opt/MaSP/
RUN git checkout ${MaSP_VERSION}

COPY preprocess.sh requirements.txt ./
RUN pip install -r requirements.txt
RUN python3 -m spacy download en

# ensures python sys.std* encoding is always utf-8
ENV PYTHONIOENCODING=UTF-8