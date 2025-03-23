# Galaxy zoo classifier

- Moreno Triana Jhon Sebastián
- Nisha
- Tica Christian

## Index

- [Tasks and workflow](#task-and-workflow)
    - [Team Contributions](#team-contributions)
- [How to get the data](#how-to-get-the-data)
- [About the data](#about-the-data)
- [Project architecture](#project-architecture)
- [Set up the conda environment](#set-up-the-conda-environment)

## Task and workflow

To train a classification model and predict the labels of the images, we follow these steps:

1. **Random Image Selection**
   - We start by randomly selecting subset of image from the image dataset.

2. **Convert Images into Tabular Format**
   - Convert arrays of pixels into tabular data by treating each pixel as a feature column, with pixel intensity values as the data points.

3. **EDA and Feature Preprocessing**
   - Perform Exploratory Data Analysis (EDA) to understand the dataset distribution, check for missing values, normalize pixel values.

4. **Dimensionality Reduction**
   - Test how much we can reduce the dimensionality of the problem using one of the algorithms (PCA, kPCA, etc.).

5. **Symmetry Estimation**
   - Estimate the symmetry of the preprocessed images with respect to **12 axes** and add this information to the dataset.

6. **Cluster Analysis**
   - Determine how many clusters can be associated with the joint distribution of the data points using **t-SNE** or **UMAP**.

7. **Model Selection**
   - Build the classifier using **Random Forest** (experimenting with different tree depths and numbers) or **SVC** (Support Vector Classifier).

8. **Model Training**
   - Train the classifier on the processed dataset.

9. **Prediction**
   - Use the trained classifier to predict class labels for the test images.

### Team Contributions

> **Tica Christian - Pixel-to-Tabular Conversion**

Tica Christian was responsible for converting the raw image data into a structured tabular dataset. Each image was flattened into individual pixel intensity values, treating each pixel as a separate feature. This transformation allows us to apply machine learning techniques directly to numerical data instead of raw images.

> **Nisha - EDA, Preprocessing, and Dimensionality Reduction**

Nisha conducted Exploratory Data Analysis (EDA) to understand the distribution of pixel intensities, detect missing values, and normalize or standardize features where necessary. The preprocessing steps included:

> **Encoding categorical variables to make them suitable for machine learning.**

> **Applying VarianceThreshold to remove low-variance features.**

> **Computing Mutual Information Scores to evaluate feature importance.**

> **Applying PCA to reduce the dataset's dimensionality while preserving meaningful variance.**

> **Moreno Triana Jhon Sebastián - Symmetry Estimation**

Moreno Triana Jhon Sebastián estimated the **symmetry of each image across 12 different axes**. This involved analyzing how similar an image is when mirrored along different orientations.


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
rsync -av --progress --ignore-existing --files-from=file_list.txt --no-relative leonardo:/ ./data/images
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
