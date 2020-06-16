import numpy as np
from face_embeddings import extract_face_embeddings
from face_detector import detect_faces
from db import add_embeddings
import dlib

shape_predictor = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Path to dataset to enroll", required=True)
    parser.add_argument("-e", "--embeddings", help="Path to save embeddings",
                    default="tmp/face_embeddings.npy")
    parser.add_argument("-l", "--labels", help="Path to save labels",
                    default="tmp/labels.pkl")
    return parser.parse_args()

def get_centered_face_index(image, faces):
    img_cx = image.shape[1] // 2
    center_dist_list = np.array([(face.right()+face.left())//2 - img_cx for face in faces])
    return np.argmin(center_dist_list)

def enroll_face(image, label,
                embeddings_path="face_embeddings.npy",
                labels_path="labels.pkl",
                down_scale=1.0):

    faces = detect_faces(image, down_scale)
    if len(faces) == 1:
        face = faces[0]
    if len(faces) < 1:
        print("[!] Skipping. No faces detected.")
        return False
    if len(faces) > 1:
        #TODO: verify with g.t if taking multiple faces increases performance
        # face = faces[get_centered_face_index(image, faces)]
        print("[!] Skipping. Multiple faces detected.")
        return False

    face_embeddings = extract_face_embeddings(image, face, shape_predictor,
                                              face_recognizer)
    add_embeddings(face_embeddings, label, embeddings_path=embeddings_path,
                   labels_path=labels_path)
    return True

if __name__ == "__main__":
    import cv2
    import glob

    args = parse_args()
    #TODO: make it work with gif files
    filetypes = ["png", "jpg"]
    dataset = args.dataset.rstrip("/")
    imPaths = []

    for filetype in filetypes:
        imPaths += glob.glob("{}/*/*.{}".format(dataset, filetype))

    for path in imPaths:
        label = path.split("/")[-2]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if not enroll_face(image, label, embeddings_path=args.embeddings, labels_path=args.labels):
            print(path)