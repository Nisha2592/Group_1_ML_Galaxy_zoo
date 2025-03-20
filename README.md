# Galaxy zoo classifier

- Moreno Triana Jhon Sebastián
- Nisha
- Tica Christian

## Index

- [How to get the data](#how-to-get-the-data)
- [About the data](#about-the-data)

## How to get the data 

> [!CAUTION]
> This is going to work only if you have connection with Leonardo cluster and the data exists on the same path inside of the `file_list.txt` file

We want to get a subset of the data from the path using the `scp` command from the command line. At the same level of this `README.md` exist a file called `file_list.txt` that contains a random list of a constant number of images from the data folder in leonardo. In order to get the data you need to create a folder called `data/` and inside a folder called `images/`, should look like the following diagram:

```
./
├── README.md
├── file_list.txt
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



**The images folder**

Inside the `images/` folder exists a bunch of images of diferent kinds of galaxies, we need to analyse that images in order to train the Machine Learning model. Each image has a name that looks like this `<number>.jpg`, this number is just a label given to each image, the data related with each image is inside of the `.csv` files.


**The gz2_filename_mapping.csv file**

The `gz2_filename_mapping.csv` is a file that contains three columns that are:

- objid: the Data Release 7 (DR7) object ID for each galaxy. This should match the first column in Table 1.
- sample: string indicating the subsampling of the galaxy.  
- asset_id: an integer that corresponds to the filename of the image in the zipped file linked above.

As an example row:

587722981742084144,original,16

means that the image of the galaxy with objid 58772298174208414 is the file named `16.jpg` inside `images` folder.


**The zoo2MainSpecz.csv file**

This file contains the data related with each galaxy, however in this project we are interested in just two columns, first, the one with the objid, named `dr7objid`, and the one with the class labels, called `gz2class`. Whit this 2 .csv we can have the data and the labels for the training.
