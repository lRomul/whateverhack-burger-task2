#!/bin/bash

REL_PATH_TO_SCRIPT=$(dirname "${BASH_SOURCE[0]}")
cd "${REL_PATH_TO_SCRIPT}"
NAME="hack2"
IMAGENAME="${NAME}"
CONTNAME="--name=${NAME}"
NET="--net=host"
IPC="--ipc=host"

VOLUMES="-v $(pwd)/src:/src -v $(pwd)/data:/data"
