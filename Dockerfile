ARG COMMON_IMAGE=
FROM ${COMMON_IMAGE}
MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

RUN pip3 install -U pip

WORKDIR /opt/genienlp/
ARG GENIENLP_VERSION=master
RUN pip3 uninstall bootleg dialogues -y
RUN git fetch && git checkout ${GENIENLP_VERSION} && pip3 install -e .
RUN python3 -m spacy download en_core_web_sm

# for occasional plotting in genienlp
RUN pip3 install matplotlib~=3.0 seaborn~=0.9

ARG ADD_APEX=
RUN echo ${ADD_APEX}
RUN if [ ${ADD_APEX} == true ]; then \
		yum install -y \
			cuda-nvml-dev-$CUDA_PKG_VERSION \
			cuda-command-line-tools-$CUDA_PKG_VERSION \
			cuda-libraries-dev-$CUDA_PKG_VERSION \
			cuda-minimal-build-$CUDA_PKG_VERSION \
			libcublas-devel-10.2.2.89-1 \
			&& \
			rm -rf /var/cache/yum/* ; \
		git clone https://github.com/NVIDIA/apex /opt/apex/ ; \
		cd /opt/apex/ ; \
		pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ ; \
	fi

# add user genie-toolkit
RUN useradd -ms /bin/bash genie-toolkit

RUN mkdir /opt/genie-toolkit/ && chown genie-toolkit:genie-toolkit /opt/genie-toolkit

# npm *really* does not like running as root, and will misbehave badly when
# run as root, so we run it as a separate user
USER genie-toolkit

ARG GENIE_VERSION=master
USER genie-toolkit
RUN git clone https://github.com/stanford-oval/genie-toolkit /opt/genie-toolkit/
WORKDIR /opt/genie-toolkit/
RUN git checkout ${GENIE_VERSION}
RUN npm ci

USER root
# normally, this would be done by npm link, but when running as root, npm
# link will mess up everything because it will rerun "npm install", which
# will undo the build step we just did, so we open-code npm link ourselves
RUN rm -f /usr/local/bin/genie && \
   rm -f /usr/local/lib/node_modules/genie-toolkit && \
   ln -s /opt/genie-toolkit /usr/local/lib/node_modules/genie-toolkit && \
   ln -s /opt/genie-toolkit/dist/tool/genie.js /usr/local/bin/genie && \
   chmod +x /usr/local/bin/genie

RUN dnf -y install rsync
RUN dnf -y install wget
RUN wget https://aka.ms/downloadazcopy-v10-linux ; tar -xvf downloadazcopy-v10-linux ; cp ./azcopy_linux_amd64_*/azcopy /usr/bin/


USER genie-toolkit
COPY lib.sh sync-repos.sh ./

# Use root for now until https://github.com/aws/amazon-eks-pod-identity-webhook/issues/8 is fixed.
# There is a workaround by changing pod fsgroup but kubeflow does not provide to api to modify pod securityContext.
USER root
RUN chown genie-toolkit:genie-toolkit /opt/genie-toolkit/sync-repos.sh
WORKDIR /home/genie-toolkit
