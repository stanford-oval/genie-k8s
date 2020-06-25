ARG COMMON_IMAGE=
FROM ${COMMON_IMAGE}
MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

WORKDIR /opt/genienlp/
ARG GENIENLP_VERSION=master
RUN pip3 install --upgrade pip
RUN git fetch && git checkout ${GENIENLP_VERSION} && pip3 install -e . && pip3 install 'git+https://github.com/LiyuanLucasLiu/RAdam#egg=radam'

# uncomment the models you want to use
# RUN genienlp cache-embeddings --destdir /usr/local/share/genienlp/embeddings --embeddings bert-large-uncased-whole-word-masking
# RUN genienlp cache-embeddings --destdir /usr/local/share/genienlp/embeddings --embeddings bert-large-uncased-whole-word-masking-finetuned-squad

# uncomment it you need Apex (for mixed precision training)
# RUN yum install -y \
#        cuda-nvml-dev-$CUDA_PKG_VERSION \
#        cuda-command-line-tools-$CUDA_PKG_VERSION \
#	cuda-libraries-dev-$CUDA_PKG_VERSION \
#        cuda-minimal-build-$CUDA_PKG_VERSION \
#        libcublas-devel-10.2.2.89-1 \
#        && \
#    	rm -rf /var/cache/yum/*
# RUN git clone https://github.com/NVIDIA/apex /opt/apex/
# WORKDIR /opt/apex/
# RUN pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

ARG THINGTALK_VERSION=master
RUN git clone https://github.com/stanford-oval/thingtalk /opt/thingtalk/
WORKDIR /opt/thingtalk/
RUN git checkout ${THINGTALK_VERSION}
RUN yarn install
RUN yarn link

ARG GENIE_VERSION=master
RUN git clone https://github.com/stanford-oval/genie-toolkit /opt/genie-toolkit/
WORKDIR /opt/genie-toolkit/
RUN git checkout ${GENIE_VERSION}
RUN yarn link thingtalk
RUN yarn install

COPY lib.sh generate-dataset-job.sh train-job.sh evaluate-job.sh paraphrase-job.sh train-paraphrase-job.sh translate-job.sh ./

# add user genie-toolkit
RUN useradd -ms /bin/bash -r genie-toolkit
USER genie-toolkit
WORKDIR /home/genie-toolkit

# ensures python sys.std* encoding is always utf-8
ENV PYTHONIOENCODING=UTF-8
