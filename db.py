import numpy as np
import pickle

def add_embeddings(embedding, label,
                   embeddings_path="face_embeddings.npy",
                   labels_path="labels.pickle"):
    first_time = False
    try:
        embeddings = np.load(embeddings_path)
        labels = pickle.load(open(labels_path, 'rb'))
    except IOError:
        first_time = True

    if first_time:
        embeddings = embedding
        labels = [label]
    else:
        embeddings = np.concatenate([embeddings, embedding], axis=0)
        labels.append(label)

    np.save(embeddings_path, embeddings)
    with open(labels_path, "wb") as f:
        pickle.dump(labels, f)

    return True