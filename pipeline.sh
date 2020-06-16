#!/usr/bin/env bash

# Pipeline input variables
NAME_LIST=/mnt/Alfheim/Data/OPERA/Venezuela/summer2020/FaceID/ner_per_filtered.lst
FACEBANK_DIR=/mnt/Alfheim/Data/OPERA/Venezuela/summer2020/FaceID/facebank
FACEBANK_LOG_PATH=/mnt/Alfheim/Data/OPERA/Venezuela/summer2020/FaceID/facebank/empty_dirs.lst
REID_RESULTS_PATH=./tmp/results.npy
IMAGE_LIST=/mnt/Alfheim/Data/OPERA/Venezuela/summer2020/FaceID/image.lst
IMAGE_DIR=/home/zal/Alfheim/Data/OPERA/Venezuela/summer2020/Venezuela_Data/text/20190926125226
PIPELINE_OUTPUT_PATH=/mnt/Alfheim/Data/OPERA/Venezuela/summer2020/FaceID/output

# Download the reference images
python -u download_from_bing.py -i ${NAME_LIST} -n 5 -o ${FACEBANK_DIR}

# Validate the downloaded images
python -u validate_downloads.py -i ${FACEBANK_DIR} -l ${FACEBANK_LOG_PATH}

# First create the reference image database
python -u enroll.py -d ${FACEBANK_DIR}

# Then label all the images in image list
python -u recognize_list.py -i ${IMAGE_LIST} -o ${REID_RESULTS_PATH}

# Finally export into the desired format
python -u export_results.py -i ${REID_RESULTS_PATH} -o ${PIPELINE_OUTPUT_PATH}