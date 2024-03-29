#!/bin/bash

replace_config() {
sed \
  -e "s|@@JOB_NAME@@|${JOB_NAME}|g" \
  -e "s|@@OWNER@@|${OWNER}|g" \
  -e "s|@@IAM_ROLE@@|${IAM_ROLE}|g" \
  -e "s|@@IMAGE@@|${IMAGE}|g" \
  -e "s|@@cmdline@@|${cmdline}|g" \
  -e "s|@@GPU_NUM@@|${GPU_NUM}|g" \
  -e "s|@@GPU_TYPE@@|${GPU_TYPE}|g" \
  "$@"
}

check_config() {
  local key
  for key in $1 ; do
    if test -z "${!key}" ; then
      echo "Missing or empty configuration key ${key}" 1>&2
      exit 1
    fi
  done
}

parse_args() {
  local dollarzero argnames argspecs arg argspec argname argdefault argvalue ok
  dollarzero="$1"
  shift
  argspecs="$1"
  shift
  argnames=
  for argspec in $argspecs ; do
    # append argspecs to argnames
    case "$argspec" in
    *=*)
      # if argspec has a default value, strip the value
      argnames="$argnames "$(sed -E -e 's/^([^=]*)=(.*)$/\1/g' <<<"$argspec")
      ;;
    *)
      argnames="$argnames $argspec"
      ;;
    esac
  done
  n=0

  while test $# -gt 0 ; do
    arg=$1
    shift
    n=$((n+1))

    # -- is used to separate additional_args provided by user; we do not touch them in this function
    if test "$arg" = "--" ; then
      break
    fi
    ok=0
    if test "$arg" = "--help" || test "$arg" = "-h" ; then
      echo -n "Usage: $dollarzero" 1>&2
      for argname in $argnames ; do
        echo -n " --$argname <$argname>" 1>&2
      done
      echo
      exit 0
    fi
    # loop through all argnames until a match
    for argname in $argnames ; do
      if test "$arg" = "--$argname" ; then
        argvalue=$1
        ok=1
        n=$((n+1))
        shift
        eval "$argname"='"$argvalue"'
        break
      fi
    done
    if test "$ok" = "0" ; then
      echo "Invalid command-line argument ${arg}" 1>&2
      exit 1
    fi
  done

  # ensure all argspecs have a value - either assigned or default
  for argspec in $argspecs ; do
  is_empty=0 # if this argument is explicitly set to be empty, e.g. `argument=`
    case "$argspec" in
    *=*)
      argname=$(sed -E -e 's/^([^=]*)=(.*)$/\1/g' <<<"$argspec")
      argdefault=$(sed -E -e 's/^([^=]*)=(.*)$/\2/g' <<<"$argspec")
      if test -z "${argdefault}" ; then
        echo "--${argname} is explicitly set to be empty"
        is_empty=1
      fi
      ;;
    *)
      argname="$argspec"
      argdefault=""
      ;;
    esac

    if test -z "${!argname}" ; then
      if test -z "${argdefault}" ; then
        if test "$is_empty" = "0" ; then
          echo "Missing required command-line argument --${argname}" 1>&2
          exit 1
        fi
      fi
      eval "$argname"='"$argdefault"'
    fi
  done
}

requote() {
  for arg ; do
    echo -n " \""$(sed 's/["\\]/\\\0/g' <<<"$arg")"\""
  done
}

get_make_dir() {
  PROJECT=$1
  # The convention is to have a Makefile in a PROJECT directory at the root of WORKDIR_REPO.
  # For repos with exceptions, we handle them here.
  if test -f workdir/${PROJECT}/Makefile ; then
    echo ${PROJECT}
  else
    echo .
  fi
}
