import argparse
import numpy as np
import os.path as osp
import sys
import traceback

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_list", type=str, help="Path to the the list of image files")
    parser.add_argument("-e", "--embeddings", help="Path to saved embeddings",
                    default="tmp/face_embeddings.npy")
    parser.add_argument("-l", "--labels", help="Path to saved labels",
                    default="tmp/labels.pkl")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output .npy file with results",
                    default="tmp/results.npy")
    return parser.parse_args()


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

    image_path_list = [l.strip() for l in open(args.input_list).readlines()]
    output_dict = dict()
    for idx, img_path in enumerate(tqdm(image_path_list)):
        try:
            image = cv2.imread(img_path)
            if image is None:
                output_dict[img_path] = []
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
            output_dict[osp.basename(img_path)] = face_list
        except Exception:
            sys.stderr.write("ERROR: Exception occurred while processing {0}\n".format(img_path))
            traceback.print_exc()

    np.save(args.output_path, output_dict)
