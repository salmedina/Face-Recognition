import argparse
import numpy as np
import os
import os.path as osp
from pathlib import Path
from datetime import datetime
import sys
import traceback
"""
input_dir should be video_shot_boundaries/representative_frames/
which has the following subdirectory structure:

input_dir/
├── v_8nADSv3YasBhArou/
│   ├── v_8nADSv3YasBhArou_1.png
│   ├── v_8nADSv3YasBhArou_2.png
├── v_aO7nbb3Q7ProAYnG/
│   ├── v_aO7nbb3Q7ProAYnG_10.png
│   ├── v_aO7nbb3Q7ProAYnG_11.png
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, help="Path to the the list of image files")
    parser.add_argument("-ext", "--frame_ext", type=str, default=".png", help="Video frames image extension")
    parser.add_argument("-e", "--embeddings", help="Path to saved embeddings",
                    default="tmp/face_embeddings.npy")
    parser.add_argument("-l", "--labels", help="Path to saved labels",
                    default="tmp/labels.pkl")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output .npy file with results",
                    default="tmp/results.npy")
    return parser.parse_args()


def split_frame_name(frame_name):
    sep_pos = frame_name.rfind('_')
    return frame_name[:sep_pos], frame_name[sep_pos+1:]


def get_now_str():
    return datetime.now().strftime('%d/%m/%Y %H:%M:%S')


def recognize_face(embedding, embeddings, labels, threshold=0.5):
    distances = np.linalg.norm(embeddings - embedding, axis=1)
    argmin = np.argmin(distances)
    minDistance = distances[argmin]

    if minDistance > threshold:
        label = "Unknown"
    else:
        label = labels[argmin]

    return (label, minDistance)


if __name__ == "__main__":
    import cv2
    import argparse
    from face_detector import detect_faces
    from face_embeddings import extract_face_embeddings
    import pickle
    import dlib
    from tqdm import tqdm

    args = parse_args()

    embeddings = np.load(args.embeddings)
    labels = pickle.load(open(args.labels, 'rb'))
    shape_predictor = dlib.shape_predictor("models/"
                                           "shape_predictor_5_face_landmarks.dat")
    face_recognizer = dlib.face_recognition_model_v1("models/"
                                                     "dlib_face_recognition_resnet_model_v1.dat")

    video_id_list = [f.name for f in os.scandir(args.input_dir) if f.is_dir()]

    output_dict = dict(data_dir=args.input_dir, videos=dict(), start_dt=get_now_str())
    for video_id in tqdm(video_id_list):
        video_path = Path(args.input_dir) / video_id
        if video_id not in output_dict:
            output_dict['videos'][video_id] = dict()
        for video_frame_path in video_path.glob(f'*{args.frame_ext}'):
            try:
                frame_video_id, frame_id = split_frame_name(video_frame_path.stem)
                frame_id = int(frame_id)
                if frame_video_id != video_id:
                    print('ERROR! frame does not correspond to parent dir')
                    continue
                image = cv2.imread(str(video_frame_path))
                if image is None:
                    print(f'ERROR! Unable to open video frame {video_frame_path}')
                    continue
                image_original = image.copy()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                faces = detect_faces(image)

                face_list = []
                for face in faces:
                    embedding = extract_face_embeddings(image, face, shape_predictor, face_recognizer)
                    label, score = recognize_face(embedding, embeddings, labels)
                    bbox = (face.left(), face.top(), face.right(), face.bottom())
                    if label != "Unknown":
                        face_list.append(dict(label=label,
                                             score=score,
                                             bbox=bbox,
                                             embedding=embedding))
                output_dict['videos'][video_id][frame_id] = face_list
            except Exception:
                sys.stderr.write("ERROR: Exception occurred while processing {0}\n".format(str(video_frame_path)))
                traceback.print_exc()

        np.save(args.output_path, output_dict)

    output_dict['end_dt'] = get_now_str()
    np.save(args.output_path, output_dict)
