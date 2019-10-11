ARG COMMON_IMAGE=
FROM ${COMMON_IMAGE}
MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

#RUN wget https://oval.cs.stanford.edu/releases/pldi19-artifact.tar.xz -O /tmp/pldi19-artifact.tar.xz && \
#    mkdir /opt/pldi19-artifact && mkdir /opt/parameter-datasets && \
#    tar xvf /tmp/pldi19-artifact.tar.xz -C /opt/pldi19-artifact && \
#    mv /opt/pldi19-artifact/parameter-datasets /opt/parameter-datasets/ &&
#    mv /opt/pldi19-artifact/parameter-datasets.tsv /opt/parameter-datasets/ &&
#    rm -fr /opt/pldi19-artifact && rm -fr /tmp/pldi19-artifact.tar.xz

# install packages
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

COPY lib.sh generate-dataset-job.sh train-job.sh evaluate-job.sh .

# add user genie-toolkit
RUN useradd -ms /bin/bash -r genie-toolkit
USER genie-toolkit
WORKDIR /home/genie-toolkit

