import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates


def symmetry(image_array, num_axis):
    height, width = image_array.shape

    step = np.pi / (num_axis)
    symmetries = np.zeros(num_axis)
    jump_line = (num_axis % 2 == 0)

    cos_theta = []
    sin_theta = []

    for i in range(num_axis//2):
        theta = step * i
        cos_theta.append(np.cos(theta))
        sin_theta.append(np.sin(theta))
    
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    for i in range(num_axis//2):
        new_x = ((x_coords - width/2) * cos_theta[i] - (y_coords - height/2) * sin_theta[i]) + width/2
        new_y = ((x_coords - width/2) * sin_theta[i] + (y_coords - height/2) * cos_theta[i]) + height/2

        mask = ((new_x - width//2)**2 + (new_y-height//2)**2) < (height//2)**2

        new_x = np.clip(new_x, 0, width - 1)
        new_y = np.clip(new_y, 0, height - 1)

        coords = np.vstack([new_y[mask], new_x[mask]])
        rotated_values = map_coordinates(image_array, coords, order=1, mode='nearest')

        rotated_image = np.copy(image_array)
        rotated_image[mask] = rotated_values.reshape(-1)
        rotated_image[~mask] = 0

        symmetries[2*i] = np.abs(rotated_image[:,:width//2].astype(np.int16) - rotated_image[:,:width//2 - jump_line:-1].astype(np.int16)).sum() / (height * width)
        symmetries[2*i + 1] = np.abs(rotated_image[:height//2, :].astype(np.int16) - rotated_image[:height//2 - jump_line:-1,:].astype(np.int16)).sum() / (height * width)

    return symmetries

def get_all_symmetries(images_array, axis):
    symmetries_numpy_df = np.zeros((len(images_array), axis))

    count = 0
    for image in images_array:
        symmetries_numpy_df[count] = symmetry(image, axis)
        count += 1
    return symmetries_numpy_df

