import sys
import argparse
import os
from bing_image_downloader import downloader as bing
import traceback


def main(list_path, output_dir, num_images):
    with open(list_path) as list_file:
        for search_term in [line.strip() for line in list_file.readlines()]:
            # names list might have two columns: refkb_ID<TAB>person_name
            search_term = search_term.split('\t', 1)[-1].strip()
            if os.path.exists(os.path.join(output_dir, search_term)):
                print(f'Skipping {search_term}, path exists')
                continue
            print('Search term:', search_term)
            try:
                bing.download(query=search_term, limit=num_images, adult_filter_off='off', output_dir=output_dir, timeout=30, page_counter_limit=5)
            except Exception:
                sys.stderr.write("ERROR: Exception occurred while processing name: {0}\n".format(search_term))
                traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_file', type=str, help='Text file with term list per line')
    parser.add_argument('-o', dest='output_dir', type=str, help='Output directory')
    parser.add_argument('-n', dest='num_images', type=int, default=10, help='Max num images per search term')
    args = parser.parse_args()

    main(args.input_file, args.output_dir, args.num_images)
