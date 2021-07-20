import os
import pyaudio
import wave
from tensorflow import keras
import numpy as np
from threading import Thread, enumerate
from queue import Queue

audio_queue = Queue()

model = keras.models.load_model("audio_model_bidirectional_LSTM_overlap.h5")

dataset = np.load("dataset_overlap.npy")
std_dev = np.std(dataset)
mean = np.mean(dataset)
maximum = np.max(dataset)
min = np.min(dataset)
scale = max(maximum, abs(min))


def normalize(array, std_dev, mean, scale):
    array = array / scale
    return array - mean / std_dev


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
p = pyaudio.PyAudio()
SPEAKERS = p.get_default_output_device_info()["hostApi"]  # The modified part

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_host_api_specific_stream_info=SPEAKERS,
                frames_per_buffer=CHUNK)

frames = []


def audio_listener():
    global frames
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)


def audio_normalizer():
    global frames
    add = False
    filled = 0
    window = np.zeros((1, 32, 16516))
    while True:
        l = len(frames)
        if CHUNK * l >= 16384:
            s_arr = np.zeros((1, 1, 16516), dtype=np.int16)
            cnk = frames[:16]
            print(len(cnk))
            s_arr[0, 0, :16384] = np.frombuffer(b''.join(cnk), dtype=np.int16)
            del frames[:8]
            add = True
        if filled <= 31 and add:
            window[0, filled, :] = normalize(s_arr[0, 0, :], std_dev, mean, scale)
            filled += 1
            add = False

        elif filled > 31 and add:
            window = np.hstack((np.delete(window, 0, 1), normalize(s_arr, std_dev, mean, scale)))
            audio_queue.put(window)


def audio_predictor():
    import sys
    while True:
        q_s = audio_queue.qsize()
        if q_s > 0:
            win = np.array((q_s, 32, 16516))
            for i in range(q_s):
                g = audio_queue.get()
                print(g.shape)
                win[i, :] = g[0, :]
            sys.stdout.write(str(np.argmax(model.predict(win), axis=-1)))


listener = Thread(target=audio_listener)
normalizer = Thread(target=audio_normalizer)
predictor = Thread(target=audio_predictor)
listener.start()
normalizer.start()
predictor.start()
listener.join()
normalizer.join()
predictor.join()
