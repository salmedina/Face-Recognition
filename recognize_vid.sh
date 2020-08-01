#!/bin/bash

[ $# -ne 5 ] && { echo "Usage: $0 face_embeddings_dir/ representative_frames/ output_dir/ parent_child_tab video_data.msb"; exit 1; }
face_embeddings_dir=$(readlink -ve $1) || exit 1
representative_frames_dir=$(readlink -ve $2) || exit 1
output_dir=$(readlink -f $3)
parent_child_tab=$(readlink -ve $4) || exit 1
msb_file=$(readlink -ve $5) || exit 1

source activate salvadom_dlib

set -x

wdir=$(dirname $(readlink -f $0))
cd $wdir

mkdir -p $output_dir/video_csr/
python -u recognize_video_dir.py -i $representative_frames_dir -o $output_dir/video_results.npy \
    -e $face_embeddings_dir/face_embeddings.npy -l $face_embeddings_dir/labels.pkl

python -u export_video_results.py -msb $msb_file -tab $parent_child_tab -r $output_dir/video_results.npy \
    -o $output_dir/video_csr/ -n $face_embeddings_dir/names_kb.tab
