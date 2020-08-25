from pathlib import Path
import os
import sys
import urllib.request
import urllib
import imghdr
import posixpath
import re

'''
Python api to download image form Bing.
Author: Guru Prasad (g.gaurav541@gmail.com)
'''


class Bing():

    def __init__(self):
        self.download_count = 0
        self.headers = {'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0'}

    def save_image(self, link, file_path, timeout=30):
        request = urllib.request.Request(link, None, self.headers)
        image = urllib.request.urlopen(request, timeout=timeout).read()
        if not imghdr.what(None, image):
            print('[Error]Invalid image, not saving {}\n'.format(link))
            raise
        with open(file_path, 'wb') as f:
            f.write(image)

    def download_image(self, link, query, output_dir, timeout=30):
        self.download_count += 1

        # Get the image link
        try:
            path = urllib.parse.urlsplit(link).path
            filename = posixpath.basename(path).split('?')[0]
            file_type = filename.split(".")[-1]
            if file_type.lower() not in ["jpe", "jpeg", "jfif", "exif", "tiff", "gif", "bmp", "png", "webp", "jpg"]:
                file_type = "jpg"

            # Download the image
            print("[%] Downloading Image #{} from {}".format(self.download_count, link))
            save_path = os.path.join(output_dir,
                                     "{}.{}".format(str(self.download_count).zfill(3), file_type))
            self.save_image(link, save_path, timeout)
            print("[%] File successfully saed\n")
        except Exception as e:
            self.download_count -= 1
            print("[!] Issue getting: {}\n[!] Error:: {}".format(link, e))

    def bing(self, query, limit, adlt='off', filters='', output_dir=None, timeout=30, page_counter_limit=5):
        limit = int(limit)
        page_counter = 0
        if output_dir is None:
            output_dir = os.getcwd()

        while self.download_count < limit and page_counter < page_counter_limit:
            print('\n\n[!!]Indexing page: {}\n'.format(page_counter + 1))
            # Parse the page source and download pics
            request_url = 'https://www.bing.com/images/async?q=' + urllib.parse.quote_plus(query) + '&first=' + str(
                page_counter) + '&count=' + str(limit) + '&adlt=' + adlt + '&qft=' + filters
            request = urllib.request.Request(request_url, None, headers=self.headers)
            response = urllib.request.urlopen(request, timeout=timeout)
            html = response.read().decode('utf8')
            links = re.findall('murl&quot;:&quot;(.*?)&quot;', html)

            print("[%] Indexed {} Images on Page {}.".format(len(links), page_counter + 1))
            print("\n===============================================\n")

            for link in links:
                if self.download_count < limit:
                    self.download_image(link, query, output_dir, timeout=timeout)
                else:
                    print("\n\n[%] Done. Downloaded {} images.".format(self.download_count))
                    print("\n===============================================\n")
                    break

            page_counter += 1
