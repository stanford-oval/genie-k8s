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
    # we cannot run npm as root, it will not run the build steps correctly
    # (https://github.com/npm/cli/issues/2062)
    if test `id -u` = 0 ; then
      su genie-toolkit -c "npm install"
    else
      npm install
    fi
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
    # we cannot run npm as root, it will not run the build steps correctly
    # (https://github.com/npm/cli/issues/2062)
    if test `id -u` = 0 ; then
      # also, it looks like npm will corrupt the installation of thingtalk
      # when doing "npm install" if the package was linked already so remove
      # the link first
      rm -f node_modules/thingtalk
      su genie-toolkit -c "npm install && npm link thingtalk"
    else
      npm install
      npm link thingtalk
    fi
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
    # we cannot run npm as root, it will not run the build steps correctly
    # (https://github.com/npm/cli/issues/2062)
    if test `id -u` = 0 ; then
      rm -f node_modules/thingtalk
      rm -f node_modules/genie-toolkit
      su genie-toolkit -c "npm install && npm link thingtalk && npm link genie-toolkit"
    else
      npm install
      npm link thingtalk
      npm link genie-toolkit
    fi
  fi
fi
