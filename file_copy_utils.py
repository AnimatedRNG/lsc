import os
import shutil
import random

folders = os.listdir('./imagenet/imagenet_images')
try:
    os.mkdir('imagenet/train')
    os.mkdir('imagenet/val')
except FileExistsError as e:
    pass

for folder in folders:
    qfolder = os.path.abspath(os.path.join('./imagenet/imagenet_images', folder))
    ofolder = os.path.abspath(os.path.join('./imagenet/train' if random.random() < 0.9 else 'imagenet/val', folder))
    try:
        os.mkdir(ofolder)
    except FileExistsError as e:
        pass

    for img in os.listdir(qfolder):
        shutil.copyfile(os.path.join(qfolder, img), os.path.join(ofolder, img))
