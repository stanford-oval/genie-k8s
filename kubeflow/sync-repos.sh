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
  if test -f yarn.lock ; then
    yarn install
    yarn link
  else
    npm install
  fi
fi

cd  /opt/genie-toolkit/
GENIE_HEAD=`git rev-parse HEAD`
if [ -n "${GENIE_VERSION}" ] && [ "${GENIE_VERSION}" != "${GENIE_HEAD}" ]; then
  git fetch
  git checkout ${GENIE_VERSION}
  if test -f yarn.lock ; then
    yarn link thingtalk
    yarn install
    yarn link
  else
    npm link ../thingtalk
    npm install
    npm link
  fi
fi

cd $HOME
if [ -n "${WORKDIR_REPO}" ] && [ -n "${WORKDIR_VERSION}" ]; then
  git clone $WORKDIR_REPO workdir
  cd workdir
  git checkout ${WORKDIR_VERSION}
  if test -f yarn.lock ; then
    yarn link thingtalk
    yarn link genie-toolkit
    yarn install
  elif test -f package-lock.json ; then
    npm link ../thingtalk
    npm link ../genie-toolkit
    npm install
  fi
fi
