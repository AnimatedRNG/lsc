#!/usr/bin/bash

[ ! -d 'ImageNet-Datasets-Downloader' ] && git clone https://github.com/mf1024/ImageNet-Datasets-Downloader.git
mkdir imagenet
python ./ImageNet-Datasets-Downloader/downloader.py -data_root ./imagenet -number_of_classes 10 -images_per_class 20
python file_copy_utils.py 
