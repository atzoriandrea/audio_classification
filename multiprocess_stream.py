import os
import pyaudio
import wave
from tensorflow import keras
import numpy as np
from multiprocessing import Process, Queue, Array
import multiprocessing

from queue import Queue











def audio_listener(frames_list, other=None):
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
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames_list.put(data)


def audio_normalizer(frames_list: multiprocessing.Queue, batch_queue: multiprocessing.Queue):
    def normalize(array, std_dev, mean, scale):
        array = array / scale
        return array - mean / std_dev

    dataset = np.load("dataset_overlap.npy")
    std_dev = np.std(dataset)
    mean = np.mean(dataset)
    maximum = np.max(dataset)
    min = np.min(dataset)
    scale = max(maximum, abs(min))

    add = False
    CHUNK = 1024
    filled = 0
    window = np.zeros((1, 32, 16516))
    batch = 128
    batch_matrix = np.zeros((batch, 32, 16516))
    last_added = 0
    cnk = []
    while True:
        l = frames_list.qsize()
        if CHUNK * l >= 16384:
            s_arr = np.zeros((1, 1, 16516), dtype=np.int16)
            for _ in range(8):
                cnk.append(frames_list.get())
            if len(cnk) > 8:
                s_arr[0, 0, :16384] = np.frombuffer(b''.join(cnk), dtype=np.int16)
                #del frames_list[:8]
                add = True
                del cnk[:8]
        if filled <= 31 and add:
            window[0, filled, :] = normalize(s_arr[0, 0, :], std_dev, mean, scale)
            filled += 1
            add = False

        elif filled > 31 and add:
            window = np.hstack((np.delete(window, 0, 1), normalize(s_arr, std_dev, mean, scale)))
            batch_matrix[last_added, :, :] = window[0, :, :]
            last_added = (last_added + 1) % batch
            if last_added == batch - 1:
                batch_queue.put(batch_matrix)


def audio_predictor(batch_queue, other=None):
    import sys
    model = keras.models.load_model("audio_model_bidirectional_LSTM_overlap.h5")
    while True:
        q_s = batch_queue.qsize()
        if q_s > 0:
            batch = batch_queue.get()
            print(batch.shape)
            sys.stdout.write(str(np.max(np.argmax(model.predict(batch), axis=-1))))
            del batch


if __name__ == "__main__":
    frames = multiprocessing.Queue()
    audio_queue = multiprocessing.Queue()

    listener = Process(target=audio_listener, args=(frames, None))
    normalizer = Process(target=audio_normalizer, args=(frames, audio_queue))
    predictor = Process(target=audio_predictor, args=(audio_queue, None))
    listener.start()
    normalizer.start()
    predictor.start()
    listener.join()
    normalizer.join()
    predictor.join()
