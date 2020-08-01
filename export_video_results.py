import argparse
import json
import pickle as pk
import numpy as np
import os.path as osp
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-msb', '--msb_path', type=str, help='Path to the .msb file that has video name and doc id frame level relation')
    parser.add_argument('-tab', '--tab_path', type=str, help='Path to the .tab file that has the doc id and parent relation')
    parser.add_argument('-r', '--reid_path', type=str, help='Path to the per video frame face recognition results npy')
    parser.add_argument('-o', '--output_dir', type=str, help='Path to the directory that will store the output json files')    
    return parser.parse_args()


def build_inv_dict_from_msb(msb_path):
    '''Returns an inverse index of the doc ID based on the video name'''
    msb_entries = [l.split() for l in open(msb_path).readlines()]
    video_docid_tuples = set([(e[0], e[1].split('_')[0]) for e in msb_entries])
    inv_dict = dict()
    for video_name, doc_id in video_docid_tuples:
        inv_dict[video_name] = doc_id

    return inv_dict


def build_inv_dict_from_tab(tab_path):
    '''Returns an inverse index of the parentID based on the docID'''
    tab_entries = [l.split() for l in open(tab_path).readlines()]
    # TAB fields; 0: catalog_id, 1: version, 2:parent_id, 3:child_id, etc...
    inv_dict = dict()
    for parent_id, doc_id in [(e[2], e[3]) for e in tab_entries]:
        # TODO: make more robust, for now we are just storing the first parent we see
        if doc_id not in inv_dict:
            inv_dict[doc_id] = parent_id

    return inv_dict


def build_frame(parent_id, doc_id, face_idx, frame_idx, face_list):
    frame = {}

    face = face_list[face_idx]
    name, score, bbox = face['label'], float(face['score']), face['bbox']

    kbid = f"comexkb:{name.lower().replace(' ', '_')}"

    frame['@type'] = 'entity_evidence'
    frame['component'] = 'opera.entities.visual.salvador'
    frame['@id'] = f'data:img-entity-faceid-{doc_id}-cmu-r1-{frame_idx}'
    frame['label'] = name

    cross_reference_dict = {'@type': 'db_reference',
                            'component': 'opera.entities.visual.salvador',
                            'score': score,
                            'id': f'{kbid}',
                            'canonical_name': name}
    cross_reference = [cross_reference_dict]
    frame['interp'] = {'@type': 'entity_evidence_interp',
                       'type': 'ldcOnt:PER',
                       'score': score,
                       'form': 'named',
                       'xref': cross_reference}

    frame['provenance'] = {'left': int(bbox[0]),
                           'top': int(bbox[1]),
                           'right': int(bbox[2]),
                           'bottom': int(bbox[3]),
                           '@type': 'bounding_box',
                           'reference': f'data:{doc_id}',
                           'parent_scope': f'data:{parent_id}'}
    frame['@type'] = 'entity_evidence'

    return frame


def build_doc_json(root_id, parent_id, doc_id, video_name, face_dict):
    if video_name not in face_dict['videos']:
        return {}

    frame_list = list(face_dict['videos'][video_name].keys())
    if len(frame_list) == 0:
        return dict()

    # creating output data
    data = {}
    data["@type"] = "frame_collection"
    data["@context"] = [
        "http://www.isi.edu/isd/LOOM/opera/jsonld-contexts/resources.jsonld",
        "http://www.isi.edu/isd/LOOM/opera/jsonld-contexts/ail/0.3/frames.jsonld"]


    # process metadata
    meta = {}
    meta["@type"] = "meta_info"
    meta["component"] = "opera.entities.visual.salvador"
    meta["organization"] = "CMU"
    meta["document_id"] = f'data:{doc_id}'
    meta["media_type"] = "image"
    data['meta'] = meta

    # overall info
    frames = []
    overall = {
        "@type": "document",
        "@id": f'data:{doc_id}',
        "media_type": "image",
        "root": 'data:' + root_id
    }
    frames.append(overall)

    frame_idx = 0
    detected_face = False
    for frame_idx, face_list in face_dict['videos'][video_name].items():
        for face_idx in range(len(face_list)):
            frame = build_frame(parent_id, doc_id, face_idx, frame_idx, face_list)
            if len(frame) > 0:
                frames.append(frame)
                detected_face = True

    if not detected_face:
        return {}

    data['frames'] = frames

    return data


def main(opts):
    face_dict = np.load(opts.reid_path, allow_pickle=True).item()
    videoname_docid_dict = build_inv_dict_from_msb(opts.msb_path)
    docid_parentid_dict = build_inv_dict_from_tab(opts.tab_path)

    for video_name in face_dict['videos'].keys():
        doc_id = videoname_docid_dict[video_name]
        parent_id = docid_parentid_dict[doc_id]
        root_id = parent_id
        doc_json = build_doc_json(root_id, parent_id, doc_id, video_name, face_dict)
        if not doc_json:
            continue
        output_path = osp.join(opts.output_dir, f'{doc_id}.csr.json')
        json.dump(doc_json, open(output_path, 'w'), indent=4, ensure_ascii=False)
        print(output_path)

if __name__ == '__main__':
    opts = parse_args()
    main(opts)