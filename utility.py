import numpy as np


def get_rotation_matrix(rotation):

    # x axis
    cos_x = np.cos(rotation[0])
    sin_x = np.sin(rotation[0])
    rotation_matrix_x = np.array([
        [1.,  0.,   0.],
        [0., cos_x, -sin_x],
        [0., sin_x,  cos_x]
    ])

    # y axis
    cos_y = np.cos(rotation[1])
    sin_y = np.sin(rotation[1])
    rotation_matrix_y = np.array([
        [cos_y,  0., sin_y],
        [0.,   1.,  0.],
        [-sin_y, 0., cos_y]
    ])

    # z axis
    cos_z = np.cos(rotation[2])
    sin_z = np.sin(rotation[2])
    rotation_matrix_z = np.array([
        [cos_z, -sin_z, 0.],
        [sin_z,  cos_z, 0.],
        [0.,    0., 1.]
    ])

    rotation_matrix = np.dot(
        np.dot(rotation_matrix_z, rotation_matrix_y),
        rotation_matrix_x
    )
    return rotation_matrix
