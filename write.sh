#!/bin/bash 

#rm ./data/images/*

mkdir -p ./data/images
xargs -a file_list.txt -I {} cp -n /{} ./data/images/

