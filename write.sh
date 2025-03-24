#!/bin/bash 

rm ./data/images/*

mkdir -p ./data/images
xargs -a image_paths.txt -I {} cp {} ./data/images/

