set -ex

pip3 install -U pip

cd /opt/genienlp/
GENIENLP_HEAD=`git rev-parse HEAD`
if [ -n "${GENIENLP_VERSION}" ] && [ "${GENIENLP_VERSION}" != "${GENIENLP_HEAD}" ]; then
  git fetch
  git checkout ${GENIENLP_VERSION}
  pip3 uninstall bootleg dialogues -y
  pip3 install -e .
  python3 -m spacy download en_core_web_sm
fi

cd  /opt/genie-toolkit/
GENIE_HEAD=`git rev-parse HEAD`
if [ -n "${GENIE_VERSION}" ] && [ "${GENIE_VERSION}" != "${GENIE_HEAD}" ]; then
  git fetch
  git checkout -f ${GENIE_VERSION}
  git clean -fdx
  # we cannot run npm as root, it will not run the build steps correctly
  # (https://github.com/npm/cli/issues/2062)
  if test `id -u` = 0 ; then
    su genie-toolkit -c "npm ci"
  else
    npm ci
  fi
fi

cd $HOME
if [ -n "${WORKDIR_REPO}" ] && [ -n "${WORKDIR_VERSION}" ]; then
  git clone $WORKDIR_REPO workdir
  cd workdir
  git checkout ${WORKDIR_VERSION}
  if test -f package-lock.json ; then
    # we cannot run npm as root, it will not run the build steps correctly
    # (https://github.com/npm/cli/issues/2062)
    if test `id -u` = 0 ; then
      # grant search permissions along the path, and ownership of the directory
      # we're modifying
      chmod +x $HOME $PWD
      chown -R genie-toolkit:genie-toolkit .
      su genie-toolkit -c "npm ci && npm link genie-toolkit"
    else
      npm ci
      npm link genie-toolkit
    fi
  fi
fi
