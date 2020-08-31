import os
import shutil

try:
    from bing import Bing
except ImportError:  # Python 3
    from .bing import Bing


def download(query, limit=100, adult_filter_off=True, force_replace=False, output_dir=None, timeout=30, page_counter_limit=5):

    engine = 'bing'
    if adult_filter_off:
        adult = 'off'
    else:
        adult = 'on'

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'dataset')
    query_dir = os.path.join(output_dir, query)

    if force_replace:
        if os.path.isdir(query_dir):
            shutil.rmtree(query_dir)

    # check output directory and create if necessary
    try:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    except:
        pass
    # check query directory and create if necessary
    print('Query dir: {}'.format(query_dir))
    if not os.path.isdir(query_dir):
        os.makedirs(query_dir)

    Bing().bing(query=query, limit=limit, adlt=adult, output_dir=query_dir, timeout=timeout, page_counter_limit=5)
