#!/usr/bin/env bash

source activate ~zaid/miniconda2/envs/salvadom_dlib/

# Pipeline input variables
FACEID_DIR=/mnt/Alfheim/Data/OPERA/Venezuela/summer2020/FaceID
NAME_LIST=${FACEID_DIR}/ner_per_filtered.lst
FACEBANK_DIR=${FACEID_DIR}/facebank
FACEBANK_LOG_PATH=${FACEID_DIR}/empty_dirs.lst
REID_RESULTS_PATH=./tmp/results.npy
IMAGE_LIST=${FACEID_DIR}/image.lst
DATA_DIR=/home/zal/Alfheim/Data/OPERA/Venezuela/summer2020/Venezuela_Data/text/20190926125226
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
python -u export_results.py -r ${REID_RESULTS_PATH} -d ${DATA_DIR} -o ${PIPELINE_OUTPUT_PATH}

source deactivate