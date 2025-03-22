# Galaxy zoo classifier

- Moreno Triana Jhon Sebastián
- Nisha
- Tica Christian

## Index

- [Tasks and workflow](#task-and-workflow)
- [How to get the data](#how-to-get-the-data)
- [About the data](#about-the-data)
- [Project architecture](#project-architecture)
- [Set up the conda environment](#set-up-the-conda-environment)

## Task and workflow

In order to train a classification model and predict the labels of the images we proceed doing the following steps:

1. Fetch the data from Leonardo (check [How to get the data](#how-to-get-the-data) section)
    - 
1. Convert array of pixels in rows of a tabular dataset, using single pixels as feature columns and the intensities as values measured.
1. Perform EDA and feature preprocessing.
1. Estimate the symmetry of the preprocessed images with respect to 12 axes and add this info to the original data.
1. Test how much you can reduce the dimensions of the problem with one algorithm between (PCA, kPCA ..).
1. Check how many clusters can be associated to the data points joint distribution using tSNE or UMAP.
1. Build the classifier using Random Forest (play with different depth and number of trees) or SVC.
1. Train the classifier.
1. Predict the class labels.


## How to get the data 

> [!CAUTION]
> This is going to work only if you have connection with Leonardo cluster and the data exists on the same path inside of the `file_list.txt` file

We want to get a subset of the data from the path using the `scp` command from the command line. At the same level of this `README.md` exist a file called `file_list.txt` that contains a random list of a constant number of images from the data folder in leonardo. In order to get the data you need to create a folder called `data/` and inside a folder called `images/`, should look like the following diagram:

```
./
├── README.md
├── file_list.txt
├── ...
└── data/
    └── images/
```

Then, you just need to run the following command (after login leonardo and have the certificate)

```bash
cat file_list.txt | xargs -I {} scp leonardo_alias:{} ./data/images
```

where `leonardo_alias` is the alias that you have on the `config` file inside `~/.ssh/` folder, i.e, the alias that you use to `ssh` leonardo. For getting the other 2 `.csv` files, you can execute the following commands:

```bash
scp leonardo_alias:/leonardo_work/ICT24_MHPC/data_projects/Project_1/data/zoo2MainSpecz.csv data
scp leonardo_alias:/leonardo_work/ICT24_MHPC/data_projects/Project_1/data/gz2_filename_mapping.csv data
```

> [!CAUTION]
> If you don't have connection to leonardo cluster or don't have access to the data, you need to download the data from [https://data.galaxyzoo.org/#section-0](https://data.galaxyzoo.org/#section-0).

## About the data

In our `data` we should have so far the following:

```
.data/
├── images/
│    └── *.jpg
├── gz2_filename_mapping.csv
└── zoo2MainSpecz.csv
```

The galaxy zoo project consist on a database of galaxy images from various sources for train Machine Learning models. 


> **The images folder**

Inside the `images/` folder exists a bunch of images of diferent kinds of galaxies, we need to analyse that images in order to train the Machine Learning model. Each image has a name that looks like this `<number>.jpg`, this number is just a label given to each image, the data related with each image is inside of the `.csv` files.


> **The gz2_filename_mapping.csv file**

The `gz2_filename_mapping.csv` is a file that contains three columns that are:

- objid: the Data Release 7 (DR7) object ID for each galaxy. This should match the first column in Table 1.
- sample: string indicating the subsampling of the galaxy.  
- asset_id: an integer that corresponds to the filename of the image in the zipped file linked above.

As an example row:

587722981742084144,original,16

means that the image of the galaxy with objid 58772298174208414 is the file named `16.jpg` inside `images` folder.


> **The zoo2MainSpecz.csv file**

This file contains the data related with each galaxy, however in this project we are interested in just two columns, first, the one with the objid, named `dr7objid`, and the one with the class labels, called `gz2class`. Whit this 2 .csv we can have the data and the labels for the training.

## Project architecture

For a cleaner implementation the project was implemented in a modular way. We have a main jupyter notebook where we run the functions and a `src` folder where we have the definition of a set of useful functions. 

Also we have an `assets` folder that contains all the images and auxiliary files helpful for the report. The project structure is represented in the following graph

```
./
├── Project1.ipynb
├── group_project_env.yml
├── README.md
├── file_list.txt
├── .gitignore
├── assets/
│    └── *.png
├── src/
│    ├── __init__.py
│    └── *.py
└── data/
    ├── images/
    │    └── *.jpg
    ├── gz2_filename_mapping.csv
    └── zoo2MainSpecz.csv
```

Along with the main notebook there are the `group_project_env.yml` file (check [Set up the conda environment](#set-up-the-conda-environment) section), the `file_list.txt` file and the `data/` folder (check the [How to get the data](#how-to-get-the-data) and [About the data](#about-the-data) sections), the `.gitignore` file and this README file.

## Set up the conda environment

For running the project we are using the same conda environment exported from a `.yml` file named `group_project_env.yml`, inside of the file is contained al the data related with the environment including the packages that we are using, the versions and the channels. In order to create the environment you need to run the following command

```bash
conda env create --name env_name --file=group_project_env.yml
```

after the create process you can activate it using

```bash
conda activate env_name
```

If the environment is already created with a previous version of the same `.yml` file you can update it by running 


```bash
conda env update -f group_project_env.yml --prune 
```
