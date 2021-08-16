import os
import pyaudio
import wave
from tensorflow import keras
import numpy as np
from multiprocessing import Process, Queue, Array
from queue import Queue



def audio_listener(frames_list, other=None):
    frames_list = []
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
        frames_list.append(data)

if __name__ == "__main__":
    frames = []
    audio_queue = Queue()

    listener = Process(target=audio_listener, args=(frames, None))
    listener.start()
    listener.join()