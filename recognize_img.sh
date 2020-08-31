#!/bin/bash

[ $# -ne 4 ] && { echo "Usage: $0 face_embeddings_dir/ image_files_list.txt output_dir/ parent_child_tab"; exit 1; }
face_embeddings_dir=$(readlink -ve $1) || exit 1
file_list=$2 # not using readlink because this could be a named pipe
output_dir=$(readlink -f $3)
parent_child_tab=$(readlink -ve $4) || exit 1

mkdir -p $output_dir
cat $file_list > $output_dir/image_list.txt

source activate salvadom_dlib

set -x

wdir=$(dirname $(readlink -f $0))
cd $wdir

python -u recognize_list.py -i $output_dir/image_list.txt -o $output_dir/image_results.npy \
    -e $face_embeddings_dir/face_embeddings.npy -l $face_embeddings_dir/labels.pkl

mkdir -p $output_dir/image_csr/
python -u export_results.py -r $output_dir/image_results.npy -o $output_dir/image_csr/ \
    -n $face_embeddings_dir/names_kb.tab -p $parent_child_tab
