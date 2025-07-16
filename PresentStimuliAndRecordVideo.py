import cv2
import numpy as np
import os
import PySimpleGUI as Psg
import time
from concurrent.futures import ThreadPoolExecutor


class Application:
    def __init__(self):
        font = ('System', 12)
        Psg.theme('Dark')
        self._layout = [
            [Psg.Text('Press RUN button to start!', font=font, key='instruction')],
            [Psg.Text('Camera Index:', font=font, size=(13, 1)),
             Psg.InputText(font=font, size=(5, 1), key='camera_index')],
            [Psg.Text('Video File Path:', font=font, size=(13, 1)),
             Psg.InputText(font=font, size=(30, 1), key='video_file_path'),
             Psg.FileBrowse(font=font, target='video_file_path')],
            [Psg.Text('Experiment Name:', font=font, size=(13, 1)),
             Psg.InputText(font=font, size=(30, 1), key='experiment_name')],
            [Psg.Button('RUN', font=font, pad=((15, 5), (10, 5)), key='run'),
             Psg.Button('EXIT', font=font, pad=((5, 5), (10, 5)), key='exit')]]

    def run(self):
        Psg.theme('Dark')
        window = Psg.Window('PlayAndRecord', self._layout)
        while True:
            event, value = window.read()
            if event == Psg.WIN_CLOSED or event == 'exit':
                break
            elif event == 'run':
                try:
                    camera_index = int(value['camera_index'])
                except ValueError:
                    window['instruction'].Update('Invalid camera index!')
                    continue

                try:
                    video = cv2.VideoCapture(value['video_file_path'])
                except Exception as e:
                    window['instruction'].Update('Invalid video file path!')
                    print(e)
                    continue

                experiment_name = value['experiment_name']
                save_file_name = os.path.join(os.path.dirname(os.getcwd()), experiment_name + '_result.mp4')
                if os.path.exists(save_file_name):
                    window['instruction'].Update('Experiment already exists!')
                    continue

                play_and_record = PlayAndRecord(camera_index, save_file_name)
                if play_and_record.run(video) == 0:
                    window['instruction'].Update('Completed.')
                else:
                    window['instruction'].Update('Failed!')

        window.close()


class PlayAndRecord:
    def __init__(self, camera_index, save_file_name):
        self._camera_index = camera_index
        self._save_file_name = save_file_name

        self._width = 1920
        self._height = 1080
        self._fps = 25
        self._alignment_index = cv2.imread('alignment_test.png', -1)[:, :, -1] > 10

    def run(self, video):
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        capture.set(cv2.CAP_PROP_FPS, self._fps)
        writer = cv2.VideoWriter(
            self._save_file_name,
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
            self._fps, (int(self._height), int(self._width)))

        background = np.zeros((1080, 1920, 3), dtype='uint8')
        cv2.namedWindow('this is a window', cv2.WINDOW_NORMAL)
        cv2.imshow('this is a window', background)
        cv2.waitKey(1)
        cv2.moveWindow('this is a window', 1440, 0)
        cv2.moveWindow('this is a window', 1440, 0)
        cv2.resizeWindow('this is a window', 1920, 1080)
        cv2.namedWindow('this is a monitoring window')

        while True:
            image_recorded = capture.read()[1]
            image_recorded[self._alignment_index] = [0, 0, 255]
            cv2.imshow('this is a monitoring window', image_recorded)
            cv2.imshow('this is a window', background)
            if cv2.waitKey(1) == 13:
                break

        executor = ThreadPoolExecutor(max_workers=2)
        image_recorded = capture.read()[1]
        start = time.time()
        while True:
            future = executor.submit(lambda: capture.read()[1])
            cv2.imshow('this is a monitoring window', image_recorded)
            if cv2.waitKey(1) == 27:
                break
            flag, frame = video.read()
            if not flag:
                break
            cv2.imshow('this is a window', frame[:1024])
            writer.write(image_recorded)
            image_recorded = future.result()
        print(time.time() - start, 'seconds')
        writer.release()

        while True:
            image_recorded = capture.read()[1]
            image_recorded[self._alignment_index] = [0, 0, 255]
            cv2.imshow('this is a monitoring window', image_recorded)
            if cv2.waitKey(1) == 13:
                break
            cv2.imshow('this is a window', background)

        capture.release()
        cv2.destroyAllWindows()
        return 0

    @staticmethod
    def play(video):
        flag, frame = video.read()
        if not flag:
            return 1
        cv2.imshow('this is a window', frame)
        return 0


if __name__ == '__main__':
    Application().run()
