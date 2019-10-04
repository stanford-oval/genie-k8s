ARG BASE_IMAGE=stanfordoval/decanlp:latest-cuda
FROM ${BASE_IMAGE}

MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

# install basic tools
USER root
RUN yum -y install make gettext unzip

# install nodejs 10.x and yarn
RUN curl -sL https://rpm.nodesource.com/setup_10.x | bash -
RUN curl -sL https://dl.yarnpkg.com/rpm/yarn.repo | tee /etc/yum.repos.d/yarn.repo
RUN yum -y install nodejs yarn

# install aws client
RUN pip3 install awscli

# download PPDB
RUN wget https://parmesan.stanford.edu/glove/ppdb-2.0-m-lexical.bin -O /usr/local/share/ppdb-2.0-m-lexical.bin && \
    chmod 755 /usr/local/share/ppdb-2.0-m-lexical.bin
ENV PPDB=/usr/local/share/ppdb-2.0-m-lexical.bin
ENV DECANLP_EMBEDDINGS=/usr/local/share/decanlp/embeddings

# download parameter datasets
ARG THINGPEDIA_DEVELOPER_KEY=invalid

RUN npm install -g thingpedia-cli
RUN mkdir -p /opt/parameter-datasets/data
RUN thingpedia --url https://thingpedia.stanford.edu/thingpedia --access-token invalid \
    --developer-key ${THINGPEDIA_DEVELOPER_KEY} download-string-values \
    -d /opt/parameter-datasets/data --manifest /opt/parameter-datasets/parameter-datasets.tsv
RUN thingpedia --url https://thingpedia.stanford.edu/thingpedia --access-token invalid \
    --developer-key ${THINGPEDIA_DEVELOPER_KEY} download-entity-values \
    -d /opt/parameter-datasets/data --manifest /opt/parameter-datasets/parameter-datasets.tsv --append-manifest

#RUN wget https://oval.cs.stanford.edu/releases/pldi19-artifact.tar.xz -O /tmp/pldi19-artifact.tar.xz && \
#    mkdir /opt/pldi19-artifact && mkdir /opt/parameter-datasets && \
#    tar xvf /tmp/pldi19-artifact.tar.xz -C /opt/pldi19-artifact && \
#    mv /opt/pldi19-artifact/parameter-datasets /opt/parameter-datasets/ &&
#    mv /opt/pldi19-artifact/parameter-datasets.tsv /opt/parameter-datasets/ &&
#    rm -fr /opt/pldi19-artifact && rm -fr /tmp/pldi19-artifact.tar.xz

# copy source and install packages
ARG GENIE_VERSION=wip/contextual
RUN git clone https://github.com/stanford-oval/genie-toolkit /opt/genie-toolkit/
WORKDIR /opt/genie-toolkit/
RUN git checkout ${GENIE_VERSION}
RUN yarn install

COPY generate-dataset-job.sh train-job.sh evaluate-job.sh .

# add user genie-toolkit
RUN useradd -ms /bin/bash -r genie-toolkit
USER genie-toolkit
WORKDIR /home/genie-toolkit

