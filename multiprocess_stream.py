import os
import pyaudio
import wave
from tensorflow import keras
import numpy as np
from multiprocessing import Process, Queue, Array
import multiprocessing
from pydub import AudioSegment
from queue import Queue


def audio_listener(frames_list, other=None, source="not_mic"):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    p = pyaudio.PyAudio()
    if source != "mic":
        SPEAKERS = p.get_device_info_by_index(6)["hostApi"]  # The modified part
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_host_api_specific_stream_info=SPEAKERS,
                        frames_per_buffer=CHUNK)
        print("Loopback device")
    else:
        MIC = p.get_default_input_device_info()["hostApi"]
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_host_api_specific_stream_info=MIC,
                        frames_per_buffer=CHUNK)
        print("Microphone device")
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames_list.put(data)


def audio_normalizer(frames_list: multiprocessing.Queue, batch_queue: multiprocessing.Queue):
    def normalize(array, std_dev, mean, scale):
        array = array / scale
        return array - mean / std_dev

    def match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    def scale_data(data):
        m = max(data)
        if m != 0:
            ratio = 32767 / m
            return (data * ratio).astype(np.int16)
        return data

    dataset = np.load("/media/andrea/My Passport/splitted_ds_v2/fold3/train_ds.npy")
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
        if CHUNK * l >= 16516:
            s_arr = np.zeros((1, 1, 16516), dtype=np.int16)
            for _ in range(8):
                cnk.append(frames_list.get())
            if len(cnk) > 9:
                #audio_s = AudioSegment(b''.join(cnk), channels=1, frame_rate=44100, sample_width=4)
                #audio_norm_v = match_target_amplitude(audio_s, -20.0)
                s_arr[0, 0, :16384] = scale_data(np.frombuffer(b''.join(cnk), dtype=np.int16)[:16516])#np.frombuffer(audio_norm_v.raw_data, dtype=np.int16))
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
                # np.save("/home/andrea/Scrivania/test_batch.npy", batch_matrix)
                # break


def audio_predictor(batch_queue, other=None):
    import sys
    import pickle as pk
    #model = keras.models.load_model("/media/andrea/My Passport/splitted_dataset/fold1/model1.h5")
    model = keras.models.load_model("half_model.h5")
    pca = pk.load(open("pca.pkl", 'rb'))
    clf = pk.load(open("clf.pkl", 'rb'))
    while True:
        q_s = batch_queue.qsize()
        if q_s > 0:
            batch = batch_queue.get()
            # print(batch.shape)
            res = model.predict(batch)
            pca_res = pca.transform(res)
            OCsvm_res = clf.predict(pca_res)
            print(OCsvm_res)
            # print("Anomaly") if avg > 0.7 else print("Normal")
            # print(avg)
            # sys.stdout.write(str(np.max(np.argmax(model.predict(batch), axis=-1))))
            del batch
            # break


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
