# Face Recognition Pipeline
This repository hosts a complete face recognition pipeline using dlib pre-trained models.

### Installation

Create a conda or virtualenv environment with a Python 3.7 base, and install the dependencies named in `requirements.txt`
```
conda create -n facerecog python=3.7
conda activate facerecog
pip install -r requirements.txt
```

### How to use

#### 1. Create a face databank

You need to create a face databank by putting in a directory the following structure:
```bash
person_1/
    img1.jpg
    img2.jpg
    ...
person_2/
    image1.png
    image2.png
    ...
...
```
The amount of images per person does not need to be the same. The name of the image within the folder also doesnt' matter, as long as it is a `png`, `jpg` or `gif` file.

#### 2. Enroll people into the system
To enroll all the persons in your face databank you need to call the following:

`$ python enroll.py --dataset <path to dataset>`

If an image has multiple faces, the enrolling script will consider the person's face the one who is more horizontally aligned to the center of the image.

#### 3. Recognize people
Finally to recognize people within an image you can call:

```
$ python recognize.py --i <path to image>
```

Or to recognize people from a file with the list of image paths you may call:

```
$ python recognize_list.py --i <path to image list>
```
The results will be stored in a numpy file which contains a dictionary with the results per image.
