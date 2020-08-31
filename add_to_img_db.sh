#!/bin/bash

[ $# -ne 2 ] && { echo "Usage: $0 names_kb.tab face_embeddings_dir/"; exit 1; }
names_kb=$(readlink -ve $1) || exit 1
face_embeddings_dir=$(readlink -f $2)

wdir=$(dirname $(readlink -f $0))
cd $wdir

mkdir -p $face_embeddings_dir/facebank

source activate salvadom_dlib

# Download the reference images
python -u download_from_bing.py -i $names_kb -n 5 -o $face_embeddings_dir/facebank/

# Validate the downloaded images
python -u validate_downloads.py -i $face_embeddings_dir/facebank/ -l $face_embeddings_dir/empty_dirs.lst --purge

# create the reference image database
rm -v $face_embeddings_dir/face_embeddings.npy $face_embeddings_dir/labels.pkl
python -u enroll.py -d $face_embeddings_dir/facebank/ -e $face_embeddings_dir/face_embeddings.npy -l $face_embeddings_dir/labels.pkl

old_names_kb=$face_embeddings_dir/names_kb.tab.$(date "+%Y-%m-%dT%H:%M:%S")
mv $face_embeddings_dir/names_kb.tab $old_names_kb
cat $names_kb $old_names_kb |sort -u > $face_embeddings_dir/names_kb.tab
