ARG COMMON_IMAGE=
FROM ${COMMON_IMAGE}
MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

# install packages
USER root
WORKDIR /opt/genienlp/
ARG GENIENLP_VERSION=master
RUN pip3 install --upgrade pip
RUN git fetch && git checkout ${GENIENLP_VERSION} && pip3 install -e .


# download additional embeddings
RUN genienlp cache-embeddings -d /usr/local/share/genienlp/embeddings --embeddings bert-base-multilingual-uncased



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

COPY lib.sh generate-dataset-job.sh train-job.sh evaluate-job.sh ./

# add user genie-toolkit
RUN useradd -ms /bin/bash -r genie-toolkit
USER genie-toolkit
WORKDIR /home/genie-toolkit

USER root
RUN chmod -R 755 /usr/local/share/genienlp/embeddings
# RUN ls -R -al /usr/local/share/genienlp/

USER genie-toolkit

