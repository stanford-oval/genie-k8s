set -ex

cd /opt/genienlp/
GENIENLP_HEAD=`git rev-parse HEAD`
if [ -n "${GENIENLP_VERSION}" ] && [ "${GENIENLP_VERSION}" != "${GENIENLP_HEAD}" ]; then
  git fetch
  git checkout ${GENIENLP_VERSION}
  pip3 install --upgrade --use-feature=2020-resolver -e .
fi


cd /opt/thingtalk/
THINGTALK_HEAD=`git rev-parse HEAD`
if [ -n "${THINGTALK_VERSION}" ] && [ "${THINGTALK_VERSION}" != "${THINGTALK_HEAD}" ]; then
  git fetch
  git checkout ${THINGTALK_VERSION}
  yarn install
  yarn link
fi

cd  /opt/genie-toolkit/
GENIE_HEAD=`git rev-parse HEAD`
if [ -n "${GENIE_VERSION}" ] && [ "${GENIE_VERSION}" != "${GENIE_HEAD}" ]; then
  git fetch
  git checkout ${GENIE_VERSION}
  yarn link thingtalk
  yarn install
  yarn link
fi

cd $HOME
if [ -n "${WORKDIR_REPO}" ] && [ -n "${WORKDIR_VERSION}" ]; then
  git clone $WORKDIR_REPO workdir
  cd workdir
  git checkout ${WORKDIR_VERSION}
   if [ -n "${WORKDIR_S3_CONFIG_DIR}" ]; then
     aws s3 cp --recursive ${WORKDIR_S3_CONFIG_DIR} .
   fi
  yarn link thingtalk
  yarn link genie-toolkit
  yarn
fi
