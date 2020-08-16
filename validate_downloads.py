import sys
import argparse
import os
import shutil
import traceback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, help='Directory to be verified if has empty sub-folders')
    parser.add_argument('-l', '--log_path', type=str, help='Path to the file which will store the list of dirs which were empty')
    parser.add_argument('--purge', action='store_true', help='Defines if the script must delete the empty directories')
    return parser.parse_args()


def main(opts):
    empty_dirs_list = [dirpath for (dirpath, dirnames, filenames) in os.walk(opts.input_dir) if
                       len(dirnames) == 0 and len(filenames) == 0]

    if opts.purge:
        for empty_dir_path in empty_dirs_list:
            print(empty_dir_path)
            try:
                os.rmdir(empty_dir_path)
            except Exception:
                sys.stderr.write("ERROR: Exception occurred while deleting empty dir {0}\n".format(empty_dir_path))
                traceback.print_exc()

    if len(empty_dirs_list) > 0:
        with open(opts.log_path, 'w') as log_file:
            log_file.write('\n'.join(empty_dirs_list))


if __name__ == '__main__':
    '''This script saves in a log file the list of empty directories. if purge is enabled, it deletes the empty dirs'''
    opts = parse_args()
    main(opts)
