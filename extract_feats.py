import argparse
import numpy as np
import os.path as osp
import sys
import traceback
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_list", type=str, help="Path to the the list of image files")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output .npy file with results",
                    default="tmp/results.npy")
    return parser.parse_args()


def extract_face_embeddings(image, face_rect, shape_predictor, face_recognizer):
    shape = shape_predictor(image, face_rect)
    face_embedding = face_recognizer.compute_face_descriptor(image, shape)
    face_embedding = [x for x in face_embedding]
    face_embedding = np.array(face_embedding, dtype="float32")[np.newaxis, :]
    return face_embedding


if __name__ == "__main__":
    import cv2
    import argparse
    from face_detector import detect_faces
    import dlib
    from tqdm import tqdm

    args = parse_args()

    shape_predictor = dlib.shape_predictor("models/"
                                           "shape_predictor_5_face_landmarks.dat")
    face_recognizer = dlib.face_recognition_model_v1("models/"
                                                     "dlib_face_recognition_resnet_model_v1.dat")

    image_path_list = [l.strip() for l in open(args.input_list).readlines()]
    output_dict = dict()
    for idx, img_path in enumerate(tqdm(image_path_list)):
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                output_dict[img_path] = []
                continue
            image_original = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect
            faces = detect_faces(image)

            face_list = []
            for face in faces:
                embedding = extract_face_embeddings(image, face, shape_predictor, face_recognizer)
                face_list.append(dict(bbox=(face.left(), face.top(), face.right(), face.bottom()),
                                      embedding=embedding))
            if len(face_list) < 1:
                continue

            print(Path(img_path).name, len(face_list))
            output_dict[Path(img_path).name] = dict(num_faces=len(face_list),
                                                       faces=face_list)
        except Exception:
            sys.stderr.write("ERROR: Exception occurred while processing {0}\n".format(img_path))
            traceback.print_exc()

    np.save(args.output_path, output_dict)
