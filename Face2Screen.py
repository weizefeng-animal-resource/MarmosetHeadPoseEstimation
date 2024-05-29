import numpy as np
import lmfit
from get_rotation_matrix import get_rotation_matrix


class Face2Screen:
    def __init__(self,
                 rotation_screen2camera,
                 translation_screen2camera,
                 scale_factor
                 ):
        self.rotation_screen2camera = rotation_screen2camera
        self.translation_screen2camera = translation_screen2camera
        self.scale_factor = scale_factor

    def calc_screen_coordinates(self, rotation_camera2face, translation_camera2face):
        translation_camera2face_coordinate_transformed = \
            np.einsum('jk,ik->ij', get_rotation_matrix(self.rotation_screen2camera).T, translation_camera2face) * \
            self.scale_factor
        translation_screen2face = self.translation_screen2camera + translation_camera2face_coordinate_transformed
        facial_direction = np.einsum(
            'jk,ik->ij',
            get_rotation_matrix(self.rotation_screen2camera),
            np.einsum(
                'ijk,k->ij',
                np.array([get_rotation_matrix(rotation) for rotation in rotation_camera2face]),
                np.array([0., 0., 1.])
            )
        )
        screen_coordinates = \
            (facial_direction.T[:-1] / facial_direction[:, -1] * translation_screen2face[:, -1] * -1).T + \
            translation_screen2face[:, :-1]
        return screen_coordinates

    def optimize_camera_position(self, screen_coordinates, rotations_camera2face, translations_camera2face):

        # define loss function
        def loss_function(screen_coordinates_real, rotation_camera2face, translation_camera2face,
                          r_x, r_y, r_z, t_x, t_y, t_z, s):
            self.rotation_screen2camera = np.array([r_x, r_y, r_z])
            self.translation_screen2camera = np.array([t_x, t_y, t_z])
            self.scale_factor = s
            screen_coordinates_calculated = self.calc_screen_coordinates(
                rotation_camera2face,
                translation_camera2face
            )
            return screen_coordinates_real - screen_coordinates_calculated

        # create model
        independent_vars = ['screen_coordinates_real', 'rotation_camera2face', 'translation_camera2face']
        model = lmfit.Model(loss_function, independent_vars=independent_vars)
        parameters = model.make_params()
        parameters['r_x'].set(value=self.rotation_screen2camera[0], vary=True, min=-np.inf, max=np.inf)
        parameters['r_y'].set(value=self.rotation_screen2camera[1], vary=False, min=-np.inf, max=np.inf)
        parameters['r_z'].set(value=self.rotation_screen2camera[2], vary=False, min=-np.inf, max=np.inf)
        parameters['t_x'].set(value=self.translation_screen2camera[0], vary=True, min=-np.inf, max=np.inf)
        parameters['t_y'].set(value=self.translation_screen2camera[1], vary=True, min=-np.inf, max=np.inf)
        parameters['t_z'].set(value=self.translation_screen2camera[2], vary=True, min=-np.inf, max=np.inf)
        parameters['s'].set(value=self.scale_factor, vary=False, min=-np.inf, max=np.inf)

        # optimize model
        result = model.fit(
            screen_coordinates_real=screen_coordinates,
            rotation_camera2face=rotations_camera2face,
            translation_camera2face=translations_camera2face,
            data=np.zeros(screen_coordinates.shape),
            params=parameters
        )

        self.rotation_screen2camera = np.array([result.values[key] for key in ['r_x', 'r_y', 'r_z']])
        self.translation_screen2camera = np.array([result.values[key] for key in ['t_x', 't_y', 't_z']])
        self.scale_factor = result.values['s']
