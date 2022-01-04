from typing import Mapping
import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec, GREEN_COLOR, RED_COLOR 
import time
import winsound
import numpy as np
import pyaudio
from collections import deque


def getNearestValue(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    return list[idx]


onnkai = {
    "C": 261.626,
    "D": 293.665,
    "E": 329.628,
    "F": 349.228,
    "G": 391.995,
    "A": 440.000,
    "B": 493.883
}
onnkai_value = [v for (k, v) in onnkai.items()]
SAMPLE_RATE = 8000

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

buffer = deque()
freq = 440
offset = 0

up = np.linspace(0, 1, 10)
down = np.linspace(1, 0, 10)

phase_offset = 0


def make_landmark_drawing_spec(index: int, len: int) -> Mapping[int, DrawingSpec]:
    dic = {}
    for i in range(len):
        if i == index:
            dic[i] = DrawingSpec(GREEN_COLOR)
        else:
            dic[i] = DrawingSpec(RED_COLOR)
    return dic

def callback(in_data, frame_count, time_info, status):
    global buffer, freq, offset, samples, phase_offset

    times = np.arange(1, frame_count + 1) * (1 / SAMPLE_RATE)  # + offset
    phase = times * freq * np.pi * 2 + phase_offset
    samples = 0.05 * np.sin(phase)
    offset = times[-1]
    phase_offset = phase[-1]
    f = open('myfile.csv', 'a')
    log = "\n".join(f"{time}, {sample}, {freq}, {p}" for time,
                    sample, p in zip(times, samples, phase))
    print("write log:", f.write(f"{log}\n"))
    f.close()
    print("in_data: ", in_data, "frame_count: ", frame_count,
          "time_info", time_info, "status:", status, "len ", len(samples))
    return (samples.astype(np.float32).tobytes(), pyaudio.paContinue)


# PyAudio開始
p = pyaudio.PyAudio()
# ストリームを開く
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                output=True,
                stream_callback=callback)

stream.start_stream()

is_ended = 0
offset = 0


def convert_to_hz(value):
    width = (onnkai["B"] - onnkai["C"])
    hz = onnkai["C"] + width * value
    index = getNearestValue(onnkai_value, hz)
    return int(index)

def main():
    global freq
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                length = len(handLms.landmark)
                target_index = length - 1
                mp_draw.draw_landmarks(img, handLms, landmark_drawing_spec=make_landmark_drawing_spec(target_index, length))
                hz = convert_to_hz(1 - handLms.landmark[target_index].y)
            freq = hz

        cv2.imshow("Image", img)
        cv2.waitKey(1)
