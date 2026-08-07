#!/bin/bash -e

set -e
scriptdir="$( cd "$(dirname "$0")" ; pwd -P )"
BUILD_PROFILE="Legacy"

for i in "$@"
do
case $i in
    -h|--help)
    HELP=true
    ;;
    -v|--verbose)
    VERBOSE=true
    BOOTLOADER_OPTION+="--verbose "
    ;;
    --legacy)
    BUILD_PROFILE="Legacy"
    ;;
    --rebase)
    BUILD_PROFILE="Rebase"
    ;;
    *)
    echo "${0##*/} - Unknown option $i"
    exit 1
    ;;
esac
done

if [ "$HELP" = true ]; then
  echo "Usage: ${0##*/} [options]"
  echo " "
  echo "Options:"
  echo "  -h|--help                    Show this help message"
  echo "  -v|--verbose                 Show verbose output"
  echo "  --legacy                    Use Legacy build profile (default)"
  echo "  --rebase                    Use Rebase build profile"
  exit 0
fi

#
# Determine the NuttX repository directory based on the selected profile.
#
set_nuttx_repo_dir() {
  if [ "$BUILD_PROFILE" = "Rebase" ]; then
    candidates=("$scriptdir/../Meadow.OS" "$scriptdir/../Rebase")
  else
    candidates=("$scriptdir/../Meadow" "$scriptdir/../Legacy")
  fi

  for candidate in "${candidates[@]}"; do
    if [ -d "$candidate" ]; then
      NUTTX_REPO_DIR="$(cd "$candidate" && pwd -P)"
      return
    fi
  done

  echo "${0##*/} - Neither ${candidates[0]} nor ${candidates[1]} exist for profile $BUILD_PROFILE."
  exit 2
}

set_nuttx_repo_dir

TOPDIR="$NUTTX_REPO_DIR/nuttx"

#
# Check if the shell is interactive.
#
check_if_interactive() {
  if [[ $- == *i* ]]; then
    red=`tput setaf 1`
    green=`tput setaf 2`
    reset=`tput sgr0`
  fi
}

#
#   Run the specified command and redirect the command output if not verbose.
#
run_command() {
  if $VERBOSE; then
    echo
    eval "$1"
  else
    eval "$1" &>/dev/null
  fi
}

#
#   Check if the last command was successful.
#
check_command_status() {
  exit_status=$?
  if [ $exit_status -ne 0 ]; then
    printf " ${red}error${reset}\n"
    if ! $VERBOSE; then
        printf "Re-run the script with --verbose flag to see the output.\n"
    fi
    exit 1
  else
    printf " ${green}success${reset}\n"
  fi
}

#
# The following is a work around for a git hub update that prevents any
# git commands from being run from within the /project directory.
# This issue has been caused by a git security update.  We do not need
# to do this on local machines, only when building using Docker.
#
if [[ "$scriptdir" == "/project" ]]; then
  run_command "git config --global --add safe.directory /project"
  check_command_status
fi

#
#   Check prerequisites.
#
if [ ! -d "$NUTTX_REPO_DIR/aot-support" ]; then
    echo "AOT support directory not found in $NUTTX_REPO_DIR."
    exit 2
else
  if [ ! -f "$NUTTX_REPO_DIR/aot-support/libbuiltin.a" ]; then
    run_command "make -C \"$NUTTX_REPO_DIR/aot-support\""
    check_command_status
  fi
fi

#
#   Run everything from the Source directory.
#
run_command "cd $scriptdir/Source"
check_command_status

#
#   Always remove the last build as there are cases where the previous build
#   causes some files not to be built.
#
run_command "make -j8 clean TOPDIR=\"$TOPDIR\""
check_command_status

#
#   Now build the system.
#
run_command "make -j8 TOPDIR=\"$TOPDIR\" $MAKE_OPTIONS"
check_command_status
