#!/usr/bin/env python
# coding: utf-8

# # Galaxy zoo classifier
# 
# <a href = "https://github.com/Jh0mpis"><img src = "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/18cef78d-8d34-4cfb-b9c7-662588f56c7a/de5p4qp-e0a4b0c8-e797-4bbf-8b4a-4d5523871a2a.jpg/v1/fill/w_1280,h_1768,q_75,strp/guts___berserker_armor_by_stephane_piovan_draw_de5p4qp-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9MTc2OCIsInBhdGgiOiJcL2ZcLzE4Y2VmNzhkLThkMzQtNGNmYi1iOWM3LTY2MjU4OGY1NmM3YVwvZGU1cDRxcC1lMGE0YjBjOC1lNzk3LTRiYmYtOGI0YS00ZDU1MjM4NzFhMmEuanBnIiwid2lkdGgiOiI8PTEyODAifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6aW1hZ2Uub3BlcmF0aW9ucyJdfQ.MsRnw-FYOYxWdtgJBJQktAgxvPV3hWjwMsBMY7-1Q5A" width="40rm"></a> **Moreno Triana Jhon Sebastián**
# 
# <a href = "https://github.com/Nisha2592">**Nisha**</a> 
# 
# <a href = "https://github.com/cristiano-mhpc" ><img src = "https://icons.iconarchive.com/icons/hektakun/pokemon/72/006-Charizard-icon.png" width="40rm"></a> **Tica Christian**
# 
# ---
# 
# The project aims to get a Random Forest classifier model to classify galxy images from the [Galaxy Zoo 2 project](https://data.galaxyzoo.org/#section-7) by doing all the usual steps on a machine learning (ML) project.
# 
# 1. [**Random Image Selection** (Moreno Triana Jhon Sebastián)](#Random-Image-Selection)
#    - We start by randomly selecting subset of image from the image dataset.
# 
# 1. [**Convert Images into Tabular Format** (Tica Christian)](#Convert-Images-into-Tabular-Format)
#    - Convert arrays of pixels into tabular data by treating each pixel as a feature column, with pixel intensity values as the data points.
# 
# 3. [**EDA and Feature Preprocessing** (Nisha)](#Perform-EDA-and-feature-preprocessing)
#    - Perform Exploratory Data Analysis (EDA) to understand the dataset distribution, check for missing values, normalize pixel values.
# 
# 5. [**Symmetry Estimation** (Moreno Triana Jhon Sebastián)](#Symmetry-Estimation)
#    - Estimate the symmetry of the preprocessed images with respect to **12 axes** and add this information to the dataset.
#   
# 1. [**Noise Reduction** (Moreno Triana Jhon Sebastián)](#Noise-Reduction)
#     - Analyzing the images we can conclude that we are interested on the brightest region, the we can depreciate the darkest areas and help to reduce the dimensionality.
# 
# 4. [**Dimensionality Reduction** (Nisha)](#Dimensionality-Reduction)
#    - Test how much we can reduce the dimensionality of the problem using one of the algorithms (PCA, kPCA, etc.).
# 
# 6. [**Cluster Analysis** (Tica Christian)](#Cluster-Analysis)
#    - Determine how many clusters can be associated with the joint distribution of the data points using **t-SNE** or **UMAP**.
# 
# 7. [**Model Selection and Model Training** (All the team)](#Model-Selection-and-Model-Training)
#    - Build the classifier using **Random Forest** (experimenting with different tree depths and numbers).
#    - Train the classifier on the processed dataset.
# 
# 9. [**Prediction**(All the team)](#Prediction)
#    - Use the trained classifier to predict class labels for the test images.
# 
# > <font color='red'><b>!IMPORTANT</b></font><br>
# > <font color='red'>We suggest you to read the [README.md](./README.md) file that is with this file before you run the cells.</font>
# 
# 
# > <font color='red'><b>!IMPORTANT</b></font><br>
# > <font color='red'>If you don't have the `file_list.txt` file please run the cells in the [Random Image Selection](#Random-Image-Selection) section.</font>

# In[1]:


#Importing libraries

import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#import Random forest classifiers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pandas as pd 


# ## Random Image Selection
# 
# Analyzing the dataset we conclude that it can be reduced to 5 classes:
# 
# - Ei:	 Eliptic galaxies of type i
# - S:	 Spiral galaxy (S + something merge on this class)
# - Er:	 Eliptic galaxies of type r
# - Ec:	 Eliptic galaxies of type c
# - SB:	 Barred spiral galaxies
# 
# And we are droping the images with label "A", there are not enough images.
# 
# Before we start to clean or extract the data we analyze the data set and get a uniform number of images per class in the dataset. This first part analyze the dataset and extract randomly a uniform set of images and writes in the `file_list.txt` file a list of files to extract from Leonardo.
# 
# > <font color='yellow'><b>ADVICE</b></font><br>
# > <font color='yellow'>If you change the number of images that you have on the `data/images/` folder you should run the following cell and run the `rsync` command again (check the [README.md](./README.md) file)</font>

# In[2]:

'''
# How the function is implemented is inside of ./src/get_files.py
from src.get_files import write_file_list

# Re-writing the file_list.txt with the new size
data_size = 4_000
write_file_list(data_size, overwrite=True)
'''

# ---

# ## Convert Images into Tabular Format

# #### Create a data frame with dr7objid and corresponding label. 
# - dr7objid gives the galaxy designation same as objid from the previous data frame.
# - label correspond to some classification of the galaxy based on its shape and morphology. 

# In[3]:


# get the objid and corresponding asset_id from gz2_filename_mapping.csv
columns_to_keep = ['objid', 'asset_id']

# Read the selected columns from the file
name_map = pd.read_csv("data/gz2_filename_mapping.csv", usecols=columns_to_keep)

# display the first few rows
# print(name_map.head(5))

#name_map.info()


# In[4]:


# select columns dr7objid and gz2class from zoo2MainSpecz.csvW
columns_to_keep = ['dr7objid', 'gz2class']

# Read the selected columns from the file
labels = pd.read_csv("data/zoo2MainSpecz.csv", usecols=columns_to_keep)

# change the name of column dr7objid to objid for merging later
labels.rename(columns={'dr7objid':'objid'}, inplace=True)

# display
#print(labels.head(5))

#labels.info()


# In[5]:


# Merge labels and name_map dataframes to map asset_id to gz2class
# merge based on objid. use an inner join (only matching rows) 
# since only a subset of points in labels are in name_map, ann inner join 
# will include the rows from name_map that have matching gz2class values
# this will avoid NaNs

labels_mapped = pd.merge(name_map, labels, on='objid', how='inner' ) 


labels_mapped.describe() # should have the same number of rows as the dataframe labels


# In[6]:


from dask import delayed, compute
import os 
from PIL import Image, ImageOps
from numpy import asarray


# In[7]:


##%%time  
# parallel implementation of processing the images with DASK

from src.data_processing import process_image_dask

borders = 100
# Directory containing images
image_dir = "data/images"

# Get list of image file paths
image_files = [
    os.path.join(image_dir, f) for f in os.listdir(image_dir)
    if f.endswith(('.png', '.jpg'))
]

# Parallel execution using Dask
delayed_results = [process_image_dask(img, borders) for img in image_files]
results = compute(*delayed_results)

# Filter out failed reads
results = [res for res in results if res is not None]

# Convert to Dask DataFrame
image_names, data = zip(*results)
image_data = pd.DataFrame(data)
image_data.insert(0, "asset_id", image_names)

#print(galaxy_data.head())

#image_data.info()

# Save to CSV
#df.to_csv("image_pixel_data.csv", index=False)
#print("Processing complete. Data saved to 'image_pixel_data.csv'.")


# In[8]:


# Merge labels_mapped with image_data to insert gz2class columnt to the latter 
# Merge based on asset_id and use an inner join. image_data which is our 
# main data frame will only have, in general, a subset of data points (galaxies)
# in labels_mapped. 
# convert asset_id values in image_data from object to int64 before mergeing

#merge
galaxy_data = pd.merge(labels_mapped, image_data, on='asset_id', how='inner') 

# Move gz2class to the last position to serve as labels
galaxy_data['gz2class'] = galaxy_data.pop('gz2class')  

# print
print(galaxy_data.head(5))

galaxy_data['gz2class'] = galaxy_data['gz2class'].str.replace(r"\bEr\S*", "Er", regex = True)
galaxy_data['gz2class'] = galaxy_data['gz2class'].str.replace(r"\bEc\S*", "Ec", regex = True)
galaxy_data['gz2class'] = galaxy_data['gz2class'].str.replace(r"\bS[a-b]\S*", "S[a-b]", regex = True)
galaxy_data['gz2class'] = galaxy_data['gz2class'].str.replace(r"\bSB[c-d]\S*", "SB[c-b]", regex = True)
galaxy_data['gz2class'] = galaxy_data['gz2class'].str.replace(r"\bEi\S*", "Ei", regex = True)
galaxy_data['gz2class'] = galaxy_data['gz2class'].str.replace(r"\bS[c-d]\S*", "S[c-d]", regex = True)

print(galaxy_data['gz2class'].value_counts())

galaxy_data.info()


# ---

# ## Perform EDA and feature preprocessing
# #### 2.1 Exploratory Data Analysis (EDA)

# In[9]:


# print
#print(galaxy_data.head(4))

print(galaxy_data.shape)  # Check dimensions

galaxy_data.info()# Check data types & missing values

# print(galaxy_data.describe())  # Get summary stats


# ---

# ## Symmetry Estimation
# 
# For this kind of images we can get the symmetry information given some axis in order to add a new data that can be relevant to the data frame. In the python module named `get_symmetry` inside of the `src/` folder we define a pair of functions that given an array of images and the number of axis of symmetry we compute the differences between the intensity values of each pixel pair.
# 
# The process of getting the symmetries is the following one:
# 
# 1. Get the coordinates of each pixel.
# 2. Rotate the coordinate system in an angle $ \theta = \dfrac{i * \pi }{n\_axis} $. Where $i$ is a number that goes from 0 to $ \dfrac{n\_axis}{2}$ and $n_axis$ is the number of axis of symmetry
# 3. Ignore the data that is outside of a circle of radius width of the image and centered in the middle of the image.
# 4. Split the image in the upper and the lower part, reflect the left part and compute the distance and append to the vector that stores the symmetry values.
# 5. Split the image in the left and right part, reflect the right part and compute the distance and append to the vector that stores the symmetry values.
# 6. Repeat until $i = \dfrac{n\_axis}{2}$.
# 7. Return the vector with the symmetry values.
# 
# For instance, an example of the step 5 is showing on the next image:
# 
# ![assets/symmetries.png](assets/symmetries.png)
# 
# Lower the distance (darker the image) more symmetric the original image.
# 
# Then, we have the following code:

# In[10]:


# Import the function from the module
from src.get_symmetry import get_all_symmetries

# Get the data related with the pixels
get_images_column = np.linspace(0, (424-2 * borders)*(424-2 * borders) - 1, (424-2 * borders)*(424-2 * borders))
# Reshape the image and assign it to an array
images_array = np.reshape(galaxy_data[get_images_column].to_numpy(), shape = (galaxy_data.shape[0], (424-2*borders), (424-2*borders)))


# In[11]:


###%%time  
from os.path import isfile
  
# Chossing the number of axis
axis = 12

# Defining the column names
columns = [f"axis-{i}" for i in range(axis)]
file_path = f'./src/{galaxy_data.shape[0]}_symmetry_{axis}_{borders}.csv'

if isfile(file_path):
    galaxy_data = pd.merge(galaxy_data, pd.read_csv(file_path), on='asset_id', how='inner')
else:
    # Getting the data
    sym_data = get_all_symmetries(images_array, axis)

    # Appending to the data frame
    for i in range(axis):
        galaxy_data[columns[i]] = sym_data[:,i]

    galaxy_data[["asset_id" ,*columns]].to_csv(file_path)

galaxy_data['gz2class'] = galaxy_data.pop('gz2class')

galaxy_data.columns = galaxy_data.columns.astype(str)

galaxy_data[["asset_id", *columns]].describe()


# After getting the previous plots we can conclude that the symmetry is not really impacting on the classification of the images. We can see that almost all the data is centered arround the same values, therefore, is not so helpfull for classify the images in this case.
# 
# ---

# ## Noise Reduction

# However, if we look at the images, we are interested on the brightest part of  it, so we can find the intensity distribution of the rows and the columns, then "_turn of_" the pixels outside $2\sigma$, where $\sigma$ is the standard deviation of the intensity distribution, and then reduce the variance on those pixels and do a Variance Threshold reduction in a further step.
# 
# We can see the difference in the following images:

# In[14]:


##%%time  
from src.remove_noise import cut_images
cut_images(galaxy_data, borders)


# In this case we can see that our revelant data is centered at the middle.

# And then, the bigest standard deviations are at the middle of the images.

# In[17]:


# In this step we check different values of variance threshold to reduce some features.
from sklearn.feature_selection import VarianceThreshold

# Drop ID and target columns first
columns_to_drop = ['objid', 'gz2class']
feature_data = galaxy_data.drop(columns=columns_to_drop)

std_pixels = np.std(galaxy_data[[str(i) for i in range((424-2*borders)*(424-2*borders))]], axis = 0).to_numpy()

print("\033[1mstd Max:\033[0m", std_pixels.max(), "\n\033[1std min:\033[0m", std_pixels.min(), "\n\033[1std mean:\033[0m", std_pixels.mean(), "\n\033[1std std:\033[0m", std_pixels.std(),"\n\033[1std mean - std:\033[0m", std_pixels.mean() - std_pixels.std())
std_mean = std_pixels.mean()


# Remove low-variance features
selector = VarianceThreshold(threshold=std_mean)
feature_data = feature_data.loc[:, selector.fit(feature_data).get_support()]

print("Shape after removing low-variance features:", feature_data.shape)
print("Number of features ignored:", galaxy_data.shape[1] - feature_data.shape[1])


# The Variance threshold has no a big effect on the features. However we saw that has a great impact on the following PCA.
# 
# ---

# ## Dimensionality Reduction

# In[18]:


##%%time  
from sklearn.decomposition import PCA

# Select only pixel features (exclude first two columns and label)
pixel_data = feature_data.iloc[:, 2:-1]  

# Standardize features: mean = 0, variance = 1
scaler = StandardScaler()
scaled_pixel_data = scaler.fit_transform(pixel_data)

# Apply PCA to retain 95% of the variance
pca = PCA(n_components=0.95)  # Retain 95% of the variance
principal_components = pca.fit_transform(scaled_pixel_data)

# Create new dataframe with principal components
galaxy_data_pca = pd.DataFrame(principal_components)

# Reinsert metadata columns
galaxy_data_pca.insert(0, "objid", galaxy_data["objid"])
galaxy_data_pca.insert(1, "asset_id", galaxy_data["asset_id"])
galaxy_data_pca["gz2class"] = galaxy_data["gz2class"]

print(f"Original shape: {galaxy_data.shape}, Reduced shape with PCA: {galaxy_data_pca.shape}")


# - PCA reduced the number of features while still keeping **$95\%$ of the variance** in the data.
# - The new reduce features are not individual pixels but **combination of pixels** that best explain the variation in the dataset.
# - Without performing the noise reduction we saw that the PCA gave us **525 features** but after the noise reduction we get just **188** features reducing by $64\%$ that have an important impact on the fitting and the prediction of the model.

# ---

# ## Cluster Analysis

# In[19]:


from sklearn.manifold import TSNE

# Convert PCA-reduced data to numpy array (excluding metadata columns)
X_pca = galaxy_data_pca.iloc[:, 2:-1].values  # Excluding "objectid", "asset_id", and "gz2class"

# Apply t-SNE to reduce to 2D
tsne = TSNE(n_components=2, perplexity=50, learning_rate=200,random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# Plot t-SNE results
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=galaxy_data_pca["gz2class"], palette="viridis", alpha=0.7)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Clustering of Galaxy Images")
plt.legend(title="Label")

# Save the figure as a JPG file
plt.savefig("tsne.jpg", format="jpg", dpi=300) 
#plt.show()


# ### **Conclusion**: Points are scattered randomly, this suggests weak separation, pixel features together with the engineered symmetry features might not be enough for a strong separation.
# 
# ---

# ## Model Selection and Model Training
# 
# To classify galaxies based on their extracted features, we implemented a **Random Forest Classifier** and experimented with different hyperparameters to optimize its performance.
# 
# #### Model Selection: Random Forest
# Random Forest is an ensemble learning method that constructs multiple decision trees and combines their outputs to improve accuracy and reduce overfitting.
# 
# #### Data Preparation
# - Used the **PCA-reduced dataset** for training.
# - Extracted features (`X`) and labels (`y`) from the dataset.
# - Split the data into **80% training** and **20% testing** sets.
# 
# 
# #### Hyperparameter Tuning
# To find the optimal model, we performed **Grid Search Cross-Validation (GridSearchCV)** with the following hyperparameters:
# - **Number of trees (`n_estimators`)**: 50, 100, 200, 300, 400
# - **Maximum tree depth (`max_depth`)**: 10, 20, 30, None
# 
# 
# #### Training the Classifier
# - Trained the **Random Forest Classifier** on the training set.
# - Used **GridSearchCV** to determine the best combination of hyperparameters.
# 
# #### Model Evaluation
# After training, the best model was selected and tested on the unseen **20% test data**. The final accuracy score was calculated to assess the performance.
# 
# 

# In[20]:


##%%time  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Load the dataset
#df = pd.read_csv("path_to_dataset.csv")  # Update with your dataset path

# Prepare the data
X = galaxy_data_pca.iloc[:, 2:-1].values
y = galaxy_data_pca["gz2class"].values

# Split the dataset into training and testing sets (80%-20% split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define different hyperparameters for experimentation
param_grid = {
'n_estimators':[200, 300, 400, 500],  # Number of trees in the forest
'max_depth': [10, 20, 30, None]  # Maximum depth of trees
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), 
    param_grid, 
    cv=3, 
    scoring='accuracy', 
    n_jobs=-1,
    verbose=2,
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_


y_pred = best_rf.predict(X_test)

# Evaluate performance
print("Best Model Accuracy:", accuracy_score(y_test, y_pred))
print("Best Hyperparameters:", grid_search.best_params_)


# In[21]:


test_classes, test_counts = np.unique(y_test, return_counts=True)
train_classes, train_counts = np.unique(y_train, return_counts=True)

print("\033[1m  Train data\t  Test data\033[0m")
print(2*"\033[1;34mClass \t\033[31mCounts\033[0m\t")
for i in range(len(train_classes)):
    print(f"\033[34m{train_classes[i]}\t\033[31m{train_counts[i]}\t\033[34m{test_classes[i]}\t\033[31m{test_counts[i]}")


# #### Results
# - **Best Model Accuracy:** `0.4766...`
# - The best-performing model was selected based on cross-validation accuracy.

# ---

# ## Prediction

# In[22]:


from sklearn.metrics import confusion_matrix
# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=set(y), yticklabels=set(y))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# Save the figure as a JPG file
plt.savefig("confusion_mat.jpg", format="jpg", dpi=300) 
#plt.show()


# The confusion Matrix shows the predictions, we can see that the class Ei has a very good chances of beeing well predicted, getting 47 out of 62 with the correct classification.
# 
# However, the other classes has a different behavior. for example the class SB has a great chance to be confussed with Er, having just 25 true predictions over 59 datapoints.
# 
# The model shows an accuracy of 48% in the best case.

# In[23]:


print(classification_report(y_test, y_pred))


# In[24]:


pred_classes, pred_counts = np.unique(y_pred, return_counts=True)
print("\033[1m  Train data\t  Test data\t  Pred data\033[0m")
print(3*"\033[1mClass \tCounts\033[0m\t")
for i in range(len(train_classes)):
    print(f"\033[1;34m{train_classes[i]}\t{train_counts[i]}\t\033[1;31m{test_classes[i]}\t{test_counts[i]}\t\033[1;32m{pred_classes[i]}\t{pred_counts[i]}")


# We can see some examples of the images and appreciate that is very difficult to classify the images with just the images, more data related with the properties of each galaxies or a better data analysis could increase the accuracy of the model.
# 
# Additionally, also we had some difficulties to increase the number of images, for instance 1500 images takes all the 16GB of RAM and we cannot run more images.
