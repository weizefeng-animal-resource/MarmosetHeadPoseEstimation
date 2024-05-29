import numpy as np
import pandas as pd
import pickle
from scipy import signal

from Camera2Face import Camera2Face
from Face2Screen import Face2Screen


def main(path):
    with open('parameters.pickle', 'br') as f:
        parameters = pickle.load(f)
    print(parameters)
    deeplabcut_output = pd.read_csv(path, header=2)

    accuracy_threshold = 0.96
    is_accurate = np.array(deeplabcut_output[[
        'likelihood',  # amount
        'likelihood.1',  # nose
        'likelihood.3', 'likelihood.5',  # right/left corner of eye
        'likelihood.6',  # center of mouth
        'likelihood.7', 'likelihood.8'  # right/left corner of mouth
    ]]) > accuracy_threshold
    face_parts_coordinates = np.array(deeplabcut_output[[
        'x', 'y',  # amount
        'x.1', 'y.1',  # nose
        'x.3', 'y.3', 'x.5', 'y.5',  # right/left corner of eye
        'x.6', 'y.6',  # center of mouth
        'x.7', 'y.7', 'x.8', 'y.8'  # right/left corner of mouth
    ]])

    crop_position = (460, 20)
    face_parts_coordinates[:, 0::2] = face_parts_coordinates[:, 0::2] + crop_position[0]
    face_parts_coordinates[:, 1::2] = face_parts_coordinates[:, 1::2] + crop_position[1]

    kernel_size = 7
    for index in range(face_parts_coordinates.shape[1]):
        face_parts_coordinates[:, index] = signal.medfilt(face_parts_coordinates[:, index], kernel_size)

    camera2face = Camera2Face(
        parameters['marmoset_face_model'],
        parameters['camera_calibration_matrix'],
        parameters['default_rotation_camera2face'],
        parameters['default_translate_camera2face']
    )

    face2screen = Face2Screen(
        parameters['default_rotation_screen2camera'],
        parameters['default_translate_screen2camera'],
        parameters['default_scale_factor'],
    )

    analysis_result = []
    for pos_frame in range(face_parts_coordinates.shape[0]):
        if np.any(np.bitwise_not(is_accurate[pos_frame])):
            analysis_result.append(np.hstack((face_parts_coordinates[pos_frame], np.zeros(8))))
        else:
            rotation, _, translation, _ = \
                camera2face.estimate_face_position(
                    face_parts_coordinates[pos_frame].reshape(-1, 2),
                    is_accurate[pos_frame]
                )
            sc = np.int64(face2screen.calc_screen_coordinates(rotation[np.newaxis, :], translation[np.newaxis, :])[0])
            analysis_result.append(np.hstack((face_parts_coordinates[pos_frame], rotation, translation, sc)))

    analysis_result = pd.DataFrame(np.array(analysis_result))
    analysis_result.columns = [
        'AmountX', 'AmountY',
        'NoseX', 'NoseY',
        'RightCornerOfEyeX', 'RightCornerOfEyeY', 'LeftCornerOfEyeX', 'LeftCornerOfEyeY',
        'CenterOfMouthX', 'CenterOfMouthY',
        'RightCornerOfMouthX', 'RightCornerOfMouthY', 'LeftCornerOfMouthX', 'LeftCornerOfMouthY',
        'RotationX', 'RotationY', 'RotationZ',
        'TranslationX', 'TranslationY', 'TranslationZ',
        'ScreenPositionX', 'ScreenPositionY'
    ]
    analysis_result.to_csv('result.csv')


if __name__ == '__main__':
    deeplabcut_output_csv = 'demo.csv'
    main(deeplabcut_output_csv)
