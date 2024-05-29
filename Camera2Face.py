import numpy as np
import lmfit
from get_rotation_matrix import get_rotation_matrix


class Camera2Face:
    def __init__(self,
                 marmoset_face_model,
                 camera_calibration_matrix,
                 default_face_rotation,
                 default_face_translation,
                 ):
        self.marmoset_face_model = marmoset_face_model
        self.camera_calibration_matrix = camera_calibration_matrix
        self.default_face_position = {
            'r_x': default_face_rotation[0], 'r_y': default_face_rotation[1], 'r_z': default_face_rotation[2],
            't_x': default_face_translation[0], 't_y': default_face_translation[1], 't_z': default_face_translation[2],
        }

    def _calc_image_coordinates(self, rotation, translation, marmoset_face_model):
        rotation_matrix = get_rotation_matrix(rotation)
        camera_coordinates = np.einsum('jk,ik->ij', rotation_matrix, marmoset_face_model) + translation
        image_coordinates = np.einsum('jk,ik->ij', self.camera_calibration_matrix, camera_coordinates)
        image_coordinates = (image_coordinates.T[:-1] / image_coordinates.T[-1]).T
        return image_coordinates

    def estimate_face_position(self, image_coordinates, is_accurate):

        # define loss function
        def loss_function(image_coordinates_real, marmoset_face_model, r_x, r_y, r_z, t_x, t_y, t_z):
            image_coordinates_calculated = self._calc_image_coordinates(
                np.array([r_x, r_y, r_z]),
                np.array([t_x, t_y, t_z]),
                marmoset_face_model
            )
            return image_coordinates_real - image_coordinates_calculated

        # create model
        model = lmfit.Model(loss_function, independent_vars=['image_coordinates_real', 'marmoset_face_model'])
        parameters = model.make_params()
        for key in parameters.keys():
            parameters[key].set(value=self.default_face_position[key], vary=True, min=-np.inf, max=np.inf)

        # optimize model
        result = model.fit(
            image_coordinates_real=image_coordinates[is_accurate],
            marmoset_face_model=self.marmoset_face_model[is_accurate],
            data=np.zeros(image_coordinates[is_accurate].shape),
            params=parameters
        )

        rotation = np.array([result.values[key] for key in ['r_x', 'r_y', 'r_z']])
        rotation_error = np.array([result.params[key].stderr for key in ['r_x', 'r_y', 'r_z']])
        translation = np.array([result.values[key] for key in ['t_x', 't_y', 't_z']])
        translation_error = np.array([result.params[key].stderr for key in ['t_x', 't_y', 't_z']])
        return rotation, rotation_error, translation, translation_error
