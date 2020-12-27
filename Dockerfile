ARG COMMON_IMAGE=
FROM ${COMMON_IMAGE}
MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

RUN pip3 install -U pip

WORKDIR /opt/genienlp/
ARG GENIENLP_VERSION=master
RUN git fetch && git checkout ${GENIENLP_VERSION} && pip3 install -e .

ARG ADD_BOOTLEG=
RUN echo ${ADD_BOOTLEG}
ARG BOOTLEG_VERSION=master
RUN if [ ${ADD_BOOTLEG} == true ]; then \
		git clone https://github.com/Mehrad0711/bootleg.git /opt/bootleg/ ; \
		cd /opt/bootleg/ ; \
		git checkout ${BOOTLEG_VERSION} && pip3 install -r requirements.txt && pip3 install -e . ; \
	fi

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
RUN if id "genie-toolkit" &>/dev/null; then \
		echo "genie-toolkit user already exists" ; \
	else \
		useradd -ms /bin/bash genie-toolkit ; \
	fi

ARG THINGTALK_VERSION=master
# npm *really* does not like running as root, and will misbehave badly when
# run as root, so we run it as a separate user

USER genie-toolkit

RUN if [ ! -d /opt/thingtalk/ ]; then \
		mkdir /opt/thingtalk/ ; \
	fi
RUN chown genie-toolkit:genie-toolkit /opt/thingtalk
RUN if [ -z "$(ls -A /opt/thingtalk/)" ]; then \
		git clone https://github.com/stanford-oval/thingtalk /opt/thingtalk/; \
	fi


RUN if [ ! -d /opt/genie-toolkit/ ]; then \
		mkdir /opt/genie-toolkit/ ; \
	fi
RUN chown genie-toolkit:genie-toolkit /opt/genie-toolkit
RUN if [ -z "$(ls -A /opt/genie-toolkit/)" ]; then \
		git clone https://github.com/stanford-oval/genie-toolkit /opt/genie-toolkit/; \
	fi

WORKDIR /opt/thingtalk/
RUN git fetch
RUN git checkout ${THINGTALK_VERSION}
RUN if test -f yarn.lock ; then \
   yarn install  ; \
   else \
   npm install ; \
   fi
USER root
# normally, this would be done by npm link, but when running as root, npm
# link will mess up everything because it will rerun "npm install", which
# will undo the build step we just did, so we open-code npm link ourselves
RUN if test -f yarn.lock ; then \
   yarn link ; \
 else \
   rm -f /usr/local/lib/node_modules/thingtalk ; \
   ln -s /opt/thingtalk /usr/local/lib/node_modules/thingtalk ; \
 fi
USER genie-toolkit

ARG GENIE_VERSION=master
WORKDIR /opt/genie-toolkit/
RUN git fetch
RUN git checkout ${GENIE_VERSION}
RUN if test -f yarn.lock ; then \
   yarn install ; \
 else \
   npm install ; \
 fi
USER root
# normally, this would be done by npm link, but when running as root, npm
# link will mess up everything because it will rerun "npm install", which
# will undo the build step we just did, so we open-code npm link ourselves
RUN if test -f yarn.lock ; then \
    yarn link thingtalk ; \
    yarn link ; \
  else \
   npm link thingtalk ; \
   rm -f /usr/local/lib/node_modules/genie-toolkit ; \
   ln -s /opt/genie-toolkit /usr/local/lib/node_modules/genie-toolkit ; \
   ln -s /opt/genie-toolkit/dist/tool/genie.js /usr/local/bin/genie ; \
   chmod +x /usr/local/bin/genie ; \
 fi

USER genie-toolkit
COPY lib.sh sync-repos.sh ./

# Use root for now until https://github.com/aws/amazon-eks-pod-identity-webhook/issues/8 is fixed.
# There is a workaround by changing pod fsgroup but kubeflow does not provide to api to modify pod securityContext.
USER root
RUN chown genie-toolkit:genie-toolkit /opt/genie-toolkit/sync-repos.sh
WORKDIR /home/genie-toolkit
