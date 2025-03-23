import numpy as np
import pandas as pd

def write_file_list(total_data, overwrite = False):
    # select columns dr7objid and gz2class from zoo2MainSpecz.csv
    columns_to_keep = ['dr7objid', 'gz2class']
    # Read the selected columns from the file
    labels = pd.read_csv("./data/zoo2MainSpecz.csv", usecols=columns_to_keep)
    # change the name of column dr7objid to objid for merging later
    labels.rename(columns={'dr7objid':'objid'}, inplace=True)
    columns_to_keep = ['objid', 'asset_id']

    # Read the selected columns from the file
    name_map = pd.read_csv("./data/gz2_filename_mapping.csv", usecols=columns_to_keep)

    labels_mapped = pd.merge(name_map, labels, on='objid', how='inner' ) 

    # Clean the data based on the class and reducing the labels
    labels_mapped = labels_mapped[labels_mapped["gz2class"] != "A"]     # Ignoring artefacts (not enough data)
    labels_mapped["gz2class"] = labels_mapped["gz2class"].str[:2].replace(
                                    r"\bS[a-z]\w*", "S", regex = True   # Replacing the words that starts with S and are followed by a lowercase letter with an S
                            )       # Reducing the labels to the first 2 letters
    labels_mapped["gz2class-numbers"] = pd.Categorical(labels_mapped["gz2class"]).codes

    # Getting the number of classes
    num_of_labels = len(labels_mapped["gz2class-numbers"].unique())
    # Size for each subset of the data
    set_size = total_data // num_of_labels
    # Get  the min size of data of a given class
    min_size_of_img = np.bincount(labels_mapped["gz2class-numbers"].to_numpy()).min()

    # The limit is to use the lower number items of a given class
    if set_size > min_size_of_img:
        set_size = min_size_of_img

    print(f"Using {set_size} images from each class")
    av_classes = labels_mapped["gz2class"].unique()
    description = {'Ei' : 'Eliptic galaxies of type i', 
                   'S'  : 'Spiral galaxy', 
                   'Er' : 'Eliptic galaxies of type r', 
                   'Ec' : 'Eliptic galaxies of type c', 
                   'SB' : 'Barred spiral galaxies'}
    print("Available classes:")
    for i in range(len(av_classes)):
        print(f"\033[1;31m\t{av_classes[i]}:\t \033[0;34m{description[av_classes[i]]}\033[0m")

    # overwrite the file_list.txt file
    if overwrite:
        # array to store the id of each image
        numbers = []

        # Generate random data, but replicable
        np.random.seed(123)
        # save the number of the random images
        for i in range(num_of_labels):
            numbers.extend(
                np.random.choice(
                    labels_mapped[labels_mapped["gz2class-numbers"] == i]["asset_id"].to_numpy(), 
                    set_size, 
                    replace = False
                )
            )

        # Path in leonardo
        folder_path = "leonardo_work/ICT24_MHPC/data_projects/Project_1/data/images/"

        # Open the file_list.txt
        file = open("file_list.txt", "w")

        # 
        for num in numbers:
            file.write(folder_path + str(num) + ".jpg\n")

        file.close()
