import argparse
import json
import pickle as pk
import numpy as np
import os.path as osp
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--names_kb', type=str, help='Path to names list (format: refkb_ID<TAB>name)')
    parser.add_argument('-p', '--parent_child', type=str, help='Path to parent_children.tab')
    parser.add_argument('-r', '--reid_path', type=str, help='Path to the per image face recognition results')
    parser.add_argument('-o', '--output_dir', type=str, help='Path to the directory that will store the output json files')    
    return parser.parse_args()

def build_frame(root_id, img_id, face_idx, frame_idx, face_list, names_kbid_dict):
    frame = {}

    face = face_list[face_idx]
    name, score, bbox = face['label'], float(face['score']), face['bbox']

    kbid = names_kbid_dict.get(name, f"comexkb:{name.lower().replace(' ', '_')}")

    frame['@type'] = 'entity_evidence'
    frame['component'] = 'opera.entities.visual.salvador'
    frame['@id'] = f'data:img-entity-faceid-{img_id}-cmu-r1-{frame_idx}'
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
                           'reference': f'data:{img_id}',
                           'parent_scope': f'data:{root_id}'}
    frame['@type'] = 'entity_evidence'

    return frame


def build_doc_json(root, img_filename, face_dict, names_kbid_dict):
    if img_filename not in face_dict:
        return dict()

    face_list = face_dict[img_filename]
    if len(face_list) == 0:
        return dict()

    img_id = osp.splitext(osp.basename(img_filename))[0]
    # creating output data
    data = {}
    data["@context"] = [
        "http://www.isi.edu/isd/LOOM/opera/jsonld-contexts/resources.jsonld",
        "http://www.isi.edu/isd/LOOM/opera/jsonld-contexts/ail/0.3/frames.jsonld"]
    data["@type"] = "frame_collection"

    # process metadata
    meta = {}
    meta["@type"] = "meta_info"
    meta["component"] = "opera.entities.visual.salvador"
    meta["organization"] = "CMU"
    meta["document_id"] = f'data:{img_id}'
    meta["media_type"] = "image"
    data['meta'] = meta

    # overall info
    frames = []
    overall = {
        "@type": "document",
        "@id": f'data:{img_id}',
        "media_type": "image",
        "root": 'data:' + root
    }
    frames.append(overall)

    frame_idx = 0
    detected_face = False
    for face_idx in range(len(face_list)):
        frame = build_frame(root, img_id, face_idx, frame_idx, face_list, names_kbid_dict)
        if len(frame) > 0:
            frames.append(frame)
            frame_idx += 1
            detected_face = True

    if not detected_face:
        return {}

    data['frames'] = frames

    return data


# TODO: replace this function with tab file and kb
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


def load_names_kb(names_kb_file):
    names_kbid_dict = {}
    if not names_kb_file:
        return names_kbid_dict
    with open(names_kb_file, 'r') as f:
        for line in f:
            fields = line.split('\t', 1)
            if len(fields) == 2:
                names_kbid_dict[fields[1].strip()] = "refkb:" + fields[0].strip()
    return names_kbid_dict


# def get_info_from_imglst(img_list_path):
#     '''Extracts from the Venezuela imgs.lst file the doc id and root'''
# 
#     img_list_name = osp.basename(img_list_path).replace('.imgs.lst', '')
#     break_pos = img_list_name.find('_')
#     root = img_list_name[:break_pos]
#     doc_id = img_list_name[break_pos+1:]
#     return root, doc_id


def main(opts):
    face_dict = np.load(opts.reid_path, allow_pickle=True).item()
    docid_parentid_dict = build_inv_dict_from_tab(opts.parent_child)
    names_kbid_dict = load_names_kb(opts.names_kb)

    # for img_list_path in glob(osp.join(opts.doc_dir, '*/*.imgs.lst')):
    #     root, doc_id = get_info_from_imglst(img_list_path)
    #     url_list = [l.strip() for l in open(img_list_path).readlines()]
    #     for url in url_list:
    #         img_filename = osp.basename(url)
    #         img_id, img_ext = osp.splitext(img_filename)
    #         doc_json = build_doc_json(root, doc_id, img_filename, face_dict, names_kbid_dict)
    #         if not doc_json:
    #             continue
    #         output_path = osp.join(opts.output_dir, f'{img_id}.csr.json')
    #         json.dump(doc_json, open(output_path, 'w'), indent=4, ensure_ascii=False)
    #         print(output_path)

    for img_filename in face_dict.keys():
        doc_id = osp.splitext(img_filename)[0]
        root = docid_parentid_dict.get(doc_id, doc_id)
        doc_json = build_doc_json(root, img_filename, face_dict, names_kbid_dict)
        if not doc_json:
            continue
        output_path = osp.join(opts.output_dir, f'{doc_id}.csr.json')
        json.dump(doc_json, open(output_path, 'w'), indent=4, ensure_ascii=False)


if __name__ == '__main__':
    opts = parse_args()
    main(opts)
