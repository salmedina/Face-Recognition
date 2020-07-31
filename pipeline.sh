#!/usr/bin/env bash

# Pipeline input variables
FACEID_DIR=/data2/salvadom/FaceID
NAME_LIST=${FACEID_DIR}/ner_per_filtered.lst
FACEBANK_DIR=${FACEID_DIR}/facebank
FACEBANK_LOG_PATH=${FACEID_DIR}/empty_dirs.lst
REID_RESULTS_PATH=./tmp/results.npy
IMAGE_LIST=${FACEID_DIR}/image.lst
DATA_DIR=/data2/OPERA/Venezuela_Data/text/20190926125226/raw
PIPELINE_OUTPUT_PATH=${FACEID_DIR}/output

# Download the reference images
python -u download_from_bing.py -i ${NAME_LIST} -n 5 -o ${FACEBANK_DIR}

# Validate the downloaded images
python -u validate_downloads.py -i ${FACEBANK_DIR} -l ${FACEBANK_LOG_PATH} --purge

# First create the reference image database
mkdir -p tmp
python -u enroll.py -d ${FACEBANK_DIR}

# Then label all the images in image list
python -u recognize_list.py -i ${IMAGE_LIST} -o ${REID_RESULTS_PATH}

# Finally export into the desired format
mkdir -p ${PIPELINE_OUTPUT_PATH}
python -u export_results.py -i ${REID_RESULTS_PATH} -d ${DATA_DIR} -o ${PIPELINE_OUTPUT_PATH}
