import numpy as np
import pandas as pd

def write_file_list(total_data, overwrite = False, Ei = False, Scd = False):
    # last 2 Optional
    final_classes = [r'\bEr\w*', r'\bEc\w*', r'\bS[a-b]\w*', r'\bSB[c-d]\w*']
    
    if Ei:
        final_classes.append(r'\bEi\w*')
    if Scd:
        final_classes.append(r'\bS[c-d]\w*')
    
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
    
    patternDel = r""
    
    for i in final_classes:
        patternDel += i + "|"
    
    class_filter = labels_mapped['gz2class'].str.contains(patternDel[:-1], regex = True)
    labels_mapped = labels_mapped[class_filter]
    labels_mapped["gz2class"] = labels_mapped["gz2class"]
    
    number_of_classes = len(final_classes)
    data_per_class = 2 *(total_data // (number_of_classes * 2))
    data_per_subclass = data_per_class//2
    
    all_posible_size = [data_per_class]
    
    all_posible_size.append(labels_mapped[labels_mapped['gz2class'].str.contains(r"\bEr", regex = True)].shape[0])
    all_posible_size.append(labels_mapped[labels_mapped['gz2class'].str.contains(r"\bEc", regex = True)].shape[0])
    all_posible_size.append(2 * labels_mapped[labels_mapped['gz2class'].str.contains(r"\bSa", regex = True)].shape[0])
    all_posible_size.append(2 * labels_mapped[labels_mapped['gz2class'].str.contains(r"\bSb", regex = True)].shape[0])
    all_posible_size.append(labels_mapped[labels_mapped['gz2class'].str.contains(r"\bSB[c-d]", regex = True)].shape[0])
    #all_posible_size.append(2 * labels_mapped[labels_mapped['gz2class'].str.contains(r"\bSBd", regex = True)].shape[0])
    
    if Ei:
        all_posible_size.append(labels_mapped['gz2class'].str.contains(r"\bEi", regex = True).shape)
    if Scd:
        all_posible_size.append(2*labels_mapped['gz2class'].str.contains(r"\bSc", regex = True).shape)
        all_posible_size.append(2*labels_mapped['gz2class'].str.contains(r"\bSd", regex = True).shape)

        
    print(all_posible_size)
    data_per_class = min(all_posible_size)
    print(all_posible_size, data_per_class)
    
    description = {'Er' : 'Eliptic galaxies of type r', 
                   'S[a-b]'  : 'Spiral galaxies of type a or b', 
                   'S[c-d]'  : 'Spiral galaxies of type c or d', 
                   'Ei' : 'Eliptic galaxies of type i', 
                   'Ec' : 'Eliptic galaxies of type c', 
                   'SB[c-d]' : 'Barred spiral galaxies of type c or d'}
    
    print(f"\033[1mUsing {data_per_class} images from each class")
    print("\033[1mAvailable classes:")
    for i in range(len(final_classes)):
        print(f"\033[1;31m\t{final_classes[i][2:-3]}:\t \033[0;34m{description[final_classes[i][2:-3]]}\033[0m")
    
    
    # overwrite the file_list.txt file
    if overwrite:
        # array to store the id of each image
        numbers = []
    
        # Generate random data, but replicable
        np.random.seed(123)
        # save the number of the random images
        numbers.extend(
            np.random.choice(
                labels_mapped[labels_mapped['gz2class'].str.contains(r"\bEr", regex = True)]["asset_id"].to_numpy(), 
                data_per_class, 
                replace = False
            )
        )
    
        numbers.extend(
            np.random.choice(
                labels_mapped[labels_mapped['gz2class'].str.contains(r"\bEc", regex = True)]["asset_id"].to_numpy(), 
                data_per_class, 
                replace = False
            )
        )
    
        numbers.extend(
            np.random.choice(
                labels_mapped[labels_mapped['gz2class'].str.contains(r"\bSa", regex = True)]["asset_id"].to_numpy(), 
                data_per_class//2, 
                replace = False
            )
        )
    
        numbers.extend(
            np.random.choice(
                labels_mapped[labels_mapped['gz2class'].str.contains(r"\bSb", regex = True)]["asset_id"].to_numpy(), 
                data_per_class//2, 
                replace = False
            )
        )
    
        numbers.extend(
            np.random.choice(
                labels_mapped[labels_mapped['gz2class'].str.contains(r"\bSB[c-d]", regex = True)]["asset_id"].to_numpy(), 
                data_per_class, 
                replace = False
            )
        )
        '''
        numbers.extend(
            np.random.choice(
                labels_mapped[labels_mapped['gz2class'].str.contains(r"\bSBd", regex = True)]["asset_id"].to_numpy(), 
                data_per_class//2, 
                replace = False
            )
        )
        '''
        
    
        if Ei:
            numbers.extend(
                np.random.choice(
                    labels_mapped[labels_mapped['gz2class'].str.contains(r"\bEi", regex = True)]["asset_id"].to_numpy(), 
                    data_per_class, 
                    replace = False
                )
            )
        if Scd:
            numbers.extend(
                np.random.choice(
                    labels_mapped[labels_mapped['gz2class'].str.contains(r"\bSc", regex = True)]["asset_id"].to_numpy(), 
                    data_per_class//2, 
                    replace = False
                )
            )
        
            numbers.extend(
                np.random.choice(
                    labels_mapped[labels_mapped['gz2class'].str.contains(r"\bSd", regex = True)]["asset_id"].to_numpy(), 
                    data_per_class//2, 
                    replace = False
                )
            )
    
    
        # Path in leonardo
        folder_path = "leonardo_work/ICT24_MHPC/data_projects/Project_1/data/images/"
    
        # Open the file_list.txt
        file = open("file_list.txt", "w")
     
        for num in numbers:
            file.write(folder_path + str(num) + ".jpg\n")
    
        file.close()

