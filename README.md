# Galaxy zoo classifier

<a href = "https://github.com/Jh0mpis"><img src = "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/18cef78d-8d34-4cfb-b9c7-662588f56c7a/de5p4qp-e0a4b0c8-e797-4bbf-8b4a-4d5523871a2a.jpg/v1/fill/w_1280,h_1768,q_75,strp/guts___berserker_armor_by_stephane_piovan_draw_de5p4qp-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9MTc2OCIsInBhdGgiOiJcL2ZcLzE4Y2VmNzhkLThkMzQtNGNmYi1iOWM3LTY2MjU4OGY1NmM3YVwvZGU1cDRxcC1lMGE0YjBjOC1lNzk3LTRiYmYtOGI0YS00ZDU1MjM4NzFhMmEuanBnIiwid2lkdGgiOiI8PTEyODAifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6aW1hZ2Uub3BlcmF0aW9ucyJdfQ.MsRnw-FYOYxWdtgJBJQktAgxvPV3hWjwMsBMY7-1Q5A" width="40rm"> </a> **Moreno Triana Jhon Sebastián**

<a href = "https://github.com/Nisha2592"><b>Nisha</b></a> 

<a href = "https://github.com/cristiano-mhpc" ><img src = "https://icons.iconarchive.com/icons/hektakun/pokemon/72/006-Charizard-icon.png" width="40rm"></a> **Tica Christian**

> [!IMPORTANT]
> If you want to check the report is on the [Project1.ipynb](Project1.ipynb) notebook


---

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
  
1. **Noise Reduction**
    - Analyzing the images we can conclude that we are interested on the brightest region, the we can depreciate the darkest areas and help to reduce the dimensionality.

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

The project aims to get a Random Forest classifier model to classify galxy images from the [Galaxy Zoo 2 project](https://data.galaxyzoo.org/#section-7) by doing all the usual steps on a machine learning (ML) project.

1. **Random Image Selection** (Moreno Triana Jhon Sebastián)
    - Jhon analyze the the data an select the subset of the images for training the model and classify.
    - Analyzing the csv related with the classes and selecting a uniform number of images of each class.
    - Set the structure of the project in order to start the project.

1. **Convert Images into Tabular Format** (Tica Christian)
    - Christian extract the image information of the images and construct the dataframes.
    - Convert arrays of pixels into tabular data by treating each pixel as a feature column, with pixel intensity values as the data points.
    - Connect each image with each classification.

3. **EDA and Feature Preprocessing** (Nisha)
    - Nisha conducted Exploratory Data Analysis (EDA) to understand the distribution of pixel intensities, detect missing values, and normalize or standardize features where necessary. The preprocessing steps included:
    - Encoding categorical variables to make them suitable for machine learning.
    - Applying VarianceThreshold to remove low-variance features.
    - Computing Mutual Information Scores to evaluate feature importance.

5. **Symmetry Estimation** (Moreno Triana Jhon Sebastián)
    - Jhon made the code for extracting the symmetry information of each image.
    - Estimate the symmetry of the preprocessed images with respect to **12 axes** and add this information to the dataset.
  
1. **Noise Reduction** (Moreno Triana Jhon Sebastián)
    - Analyzing the images we can conclude that we are interested on the brightest region, the we can depreciate the darkest areas and help to reduce the dimensionality.

4. **Dimensionality Reduction** (Nisha and Christian)
   - Test how much we can reduce the dimensionality of the problem using one of the algorithms (PCA, kPCA, etc.).

6. **Cluster Analysis** (Tica Christian)
   - Determine how many clusters can be associated with the joint distribution of the data points using **t-SNE**.

7. **Model Selection and Model Training** (All the team)}
   - Build the classifier using **Random Forest** (experimenting with different tree depths and numbers).
   - Train the classifier on the processed dataset using multiple hyperparameters.

9. **Prediction** (All the team)
   - Use the trained classifier to predict class labels for the test images.
   - Analysis of the results.

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
