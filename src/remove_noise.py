from scipy.optimize import curve_fit
import numpy as np
import pandas as pd

# Define the Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def cut_images(data_frame):
    images_columns = [str(i) for i in range(424*424)]
    numpy_df = data_frame[images_columns].to_numpy()

    images_array = numpy_df.reshape((data_frame.shape[0], 424, 424))
    index = np.linspace(0, 424 - 1, 424)

    for image in images_array:
        rows_mean_intensity = np.sum(image, axis = 1)
        cols_mean_intensity = np.sum(image, axis = 0)
        # Initial guess for parameters: (amplitude, mean, std deviation)
        initial_guess = [max(rows_mean_intensity - rows_mean_intensity.min()), np.mean(index), np.std(index)]

        try:
            # Fit the Gaussian function to the data
            params, covariance = curve_fit(gaussian, index, rows_mean_intensity - rows_mean_intensity.min(), p0=initial_guess)
            # Extract fitted parameters
            A_fit, mu_fit, sigma_fit = params      
        except:
            mu_fit = None
            sigma_fit = None

        try:
            # Initial guess for parameters: (amplitude, mean, std deviation)
            initial_guess = [max(cols_mean_intensity - cols_mean_intensity.min()), np.mean(index), np.std(index)]
            # Fit the Gaussian function to the data
            params, covariance = curve_fit(gaussian, index, cols_mean_intensity - cols_mean_intensity.min(), p0=initial_guess)
            # Extract fitted parameters
            A_fit, cols_mu_fit, cols_sigma_fit = params
        except:
            cols_mu_fit = None
            cols_sigma_fit = None

        if mu_fit and sigma_fit:
            if int(mu_fit-2*sigma_fit) > 0:
                image[:int(mu_fit-2*sigma_fit),:] = 0
            if int(mu_fit+2*sigma_fit) > 0:
                image[int(mu_fit+2*sigma_fit):,:] = 0

        if cols_mu_fit and cols_sigma_fit:
            if int(cols_mu_fit-2*cols_sigma_fit) > 0:
                image[:, :int(cols_mu_fit-2*cols_sigma_fit)] = 0
            if int(cols_mu_fit+2*cols_sigma_fit) > 0:
                image[:, int(cols_mu_fit+2*cols_sigma_fit):] = 0

    data_frame.loc[:, images_columns] = images_array.reshape(data_frame.shape[0], -1)
