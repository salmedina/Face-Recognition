#!/usr/bin/env bash

# Environment used in Rabat
# ~zaid/miniconda2/envs/salvadom_dlib/

# Pipeline input variables
FACEID_DIR=/data2/salvadom/FaceID
NAME_LIST=${FACEID_DIR}/ner_per_filtered.lst
FACEBANK_DIR=${FACEID_DIR}/facebank
FACEBANK_LOG_PATH=${FACEID_DIR}/empty_dirs.lst
# Image Processing
REID_RESULTS_PATH=./tmp/results.npy
IMAGE_LIST=${FACEID_DIR}/image.lst
DATA_DIR=/data2/OPERA/Venezuela_Data/text/20190926125226/raw
IMAGE_OUTPUT_PATH=${FACEID_DIR}/output
# Video Processing
VIDEO_REID_RESULTS_PATH=./tmp/video_results.npy
VIDEO_FRAMES_DIR=/data2/OPERA/LDC2020E11_AIDA_Phase_2_Practice_Topic_Source_Data_V1.0/data/video_shot_boundaries/representative_frames/
MSB_PATH=/data2/OPERA/LDC2020E11_AIDA_Phase_2_Practice_Topic_Source_Data_V1.0/docs/video_data.msb
TAB_PATH=/data2/OPERA/LDC2020E11_AIDA_Phase_2_Practice_Topic_Source_Data_V1.0/docs/parent_children.tab
VIDEO_OUTPUT_PATH=${FACEID_DIR}/video_output

# Download the reference images
echo "*** DOWNLOADING REFERENCE IMAGES ****************"
python -u download_from_bing.py -i ${NAME_LIST} -n 5 -o ${FACEBANK_DIR}

# Validate the downloaded images
python -u validate_downloads.py -i ${FACEBANK_DIR} -l ${FACEBANK_LOG_PATH} --purge

# First create the reference image database
echo "*** BUILDING FACEBANK ***************************"
mkdir -p tmp
python -u enroll.py -d ${FACEBANK_DIR}

# Process the images
echo "*** PROCESSING IMAGES ***************************"
python -u recognize_list.py -i ${IMAGE_LIST} -o ${REID_RESULTS_PATH}
mkdir -p ${IMAGE_OUTPUT_PATH}
python -u export_results.py -r ${REID_RESULTS_PATH} -d ${DATA_DIR} -o ${IMAGE_OUTPUT_PATH}

# Process the videos
echo "*** PROCESSING VIDEOS ***************************"
python -u recognize_video_dir.py -i ${VIDEO_FRAMES_DIR} -o ${VIDEO_REID_RESULTS_PATH}
mkdir -p ${VIDEO_OUTPUT_PATH}
python -u export_video_results.py -msb ${MSB_PATH} -tab ${TAB_PATH} -r ${VIDEO_REID_RESULTS_PATH} -o ${VIDEO_OUTPUT_PATH}