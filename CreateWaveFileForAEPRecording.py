import datetime
import numpy as np
import PySimpleGUI as Psg
import wave
import pandas as pd


class App:
    FRAME_RATE = 44100
    CHANNELS = 2
    SAMPLE_WIDTH = 2
    FREQUENCY_STANDARD = 1000
    FREQUENCY_DEVIANT = 1500
    DURATION = 0.3
    INTERVAL = 0.7
    BUFFER = 0.006
    FREQUENCY_TRIGGER = 10000
    DURATION_TRIGGER = 0.01
    NUM_LEADING_STANDARD = 4

    def __init__(self):
        self.app_name = 'SSfAEP'

        font = ('System', 12)
        Psg.theme('Dark')
        self.layout = []
        column_left = [[Psg.Text('Number of Trials', font=font)],
                       [Psg.Text('Deviant Percentage(%)', font=font)],
                       [Psg.Text('Trigger Delay(sec)', font=font)]]
        column_right = [[Psg.InputText(default_text='1000', font=font, size=(10, 1), key='iterations')],
                        [Psg.InputText(default_text='0', font=font, size=(10, 1), key='percentage')],
                        [Psg.InputText(default_text='0', font=font, size=(10, 1), key='delay')]]
        self.layout.append([Psg.Column(column_left), Psg.Column(column_right)])
        column = [[Psg.Button('RUN', font=font, button_color=('#ffffff', '#505050'), key='run'),
                   Psg.Button('EXIT', font=font, button_color=('#ffffff', '#505050'), key='exit')]]
        self.layout.append([Psg.Column(column, justification='c')])

        self.window = None

        x = np.arange(int(self.FRAME_RATE * self.DURATION))
        self.standard = np.cos(x / self.FRAME_RATE * self.FREQUENCY_STANDARD * 2 * np.pi)
        self.deviant = np.cos(x / self.FRAME_RATE * self.FREQUENCY_DEVIANT * 2 * np.pi)
        buffer = np.arange(int(self.BUFFER * self.FRAME_RATE))
        buffer = buffer / buffer[-1]
        mask = np.hstack((buffer, np.ones(x.shape[0] - buffer.shape[0] * 2), buffer[-1] - buffer))
        self.standard *= mask
        self.deviant *= mask

        self.a = np.cos(x / self.FRAME_RATE * 500 * (1.5 ** 0) * 2 * np.pi) * mask / 3.24
        self.b = np.cos(x / self.FRAME_RATE * 500 * (1.5 ** 1) * 2 * np.pi) * mask / 10.59
        self.c = np.cos(x / self.FRAME_RATE * 500 * (1.5 ** 2) * 2 * np.pi) * mask / 8.13
        self.d = np.cos(x / self.FRAME_RATE * 500 * (1.5 ** 3) * 2 * np.pi) * mask / 1
        self.e = np.cos(x / self.FRAME_RATE * 500 * (1.5 ** 4) * 2 * np.pi) * mask / 3.05

        temp = np.zeros(int(self.INTERVAL * self.FRAME_RATE))
        self.interval = np.int16(np.vstack((temp, temp)).T).copy()

        x_ = np.arange(int(self.FRAME_RATE * self.DURATION_TRIGGER))
        # self.trigger = np.cos(x_ / self.FRAME_RATE * self.FREQUENCY_TRIGGER * 2 * np.pi)
        self.trigger = np.ones(len(x_))
        self.trigger = np.hstack((self.trigger, np.zeros(x.shape[0] - x_.shape[0])))

    def run(self):
        Psg.theme('Dark')
        self.window = Psg.Window(self.app_name, self.layout)
        while True:

            event, values = self.window.read()

            if event in (Psg.WIN_CLOSED, 'exit'):
                break

            if event == 'run':

                iterations = int(values['iterations'])
                if iterations <= 0:
                    raise Exception('invalid iterations')

                deviant_percentage = float(values['percentage'])
                if deviant_percentage < 0 or deviant_percentage >= 50:
                    raise Exception('invalid deviant percentage')

                trigger_delay = float(values['delay'])
                if trigger_delay < 0 or trigger_delay >= self.DURATION - self.DURATION_TRIGGER:
                    raise Exception('invalid trigger_delay')

                delay_in_frames = int(trigger_delay * self.FRAME_RATE)
                self.trigger = np.roll(self.trigger, delay_in_frames)
                self.standard = np.int16(np.vstack((self.standard, self.trigger)).T * 250).copy()
                self.deviant = np.int16(np.vstack((self.deviant, self.trigger)).T * 250).copy()

                self.a = np.int16(np.vstack((self.a, self.trigger)).T * 30000).copy()
                self.b = np.int16(np.vstack((self.b, self.trigger)).T * 30000).copy()
                self.c = np.int16(np.vstack((self.c, self.trigger)).T * 30000).copy()
                self.d = np.int16(np.vstack((self.d, self.trigger)).T * 30000).copy()
                self.e = np.int16(np.vstack((self.e, self.trigger)).T * 30000).copy()

                count_deviants = int(iterations * deviant_percentage / 100)
                count_temp = iterations - count_deviants * 2 - self.NUM_LEADING_STANDARD
                if count_temp < 0:
                    raise Exception('invalid deviant percentage')
                flags = np.hstack((np.zeros(count_temp), np.ones(count_deviants)))
                flags = np.random.permutation(flags)
                flags = np.hstack((np.zeros(self.NUM_LEADING_STANDARD), flags))

                flags = None
                for i in range(800):
                    flag = np.random.permutation(np.arange(5))
                    if flags is None:
                        flags = flag
                    else:
                        if flags[-1] == flag[0]:
                            flag = np.roll(flag, 1)
                        flags = np.hstack((flags, flag))

                df = pd.DataFrame()
                df['flags'] = flags
                df.to_csv('tuning.csv', index=False)

                wave_file_name = \
                    str(datetime.datetime.now()).split('.')[0].replace(' ', '_') + \
                    '_deviant{0}_delay{1}ms.wav'.format(deviant_percentage / 100, trigger_delay * 1000)
                wave_file_name = 'tuning.wav'
                wave_file = wave.open(wave_file_name, 'wb')
                wave_file.setframerate(self.FRAME_RATE)
                wave_file.setnchannels(self.CHANNELS)
                wave_file.setsampwidth(self.SAMPLE_WIDTH)
                # for flag in flags:
                #     wave_file.writeframes(self.standard)
                #     wave_file.writeframes(self.interval)
                #     if flag:
                #         wave_file.writeframes(self.deviant)
                #         wave_file.writeframes(self.interval)
                for flag in flags:
                    if flag == 0:
                        wave_file.writeframes(self.a)
                        wave_file.writeframes(self.interval)
                    elif flag == 1:
                        wave_file.writeframes(self.b)
                        wave_file.writeframes(self.interval)
                    elif flag == 2:
                        wave_file.writeframes(self.c)
                        wave_file.writeframes(self.interval)
                    elif flag == 3:
                        wave_file.writeframes(self.d)
                        wave_file.writeframes(self.interval)
                    elif flag == 4:
                        wave_file.writeframes(self.e)
                        wave_file.writeframes(self.interval)

                wave_file.close()

                break


if __name__ == '__main__':
    App().run()
