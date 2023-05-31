# SOICHIRO-NISHIO-Github
# 画像を音声信号に変換し、そのスペクトログラムが元画像となるpythonコード
# img2wav.py
import cv2
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def image_to_wav(image_path, output_path, duration=10, sample_rate):
    # 画像の読み込み
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = np.flipud(image)

    # 画像のサイズ取得
    height, width = image.shape

    # スペクトログラムと、音声データ作成用のリスト
    spectrogram_data = np.zeros((height+1,width))
    audio_data = []


    # 縦列ごとに逆フーリエ変換で音声にする。
    for col in range(width):

        # 列ごとにデータを抽出
        column_data = image[:, col]
        column_data = np.append(column_data,0)

        # 逆フーリエ変換irfftを行い、音声データに変換
        tmp_audio_data = np.fft.irfft(column_data)/256
        audio_data = np.append(audio_data,tmp_audio_data,axis=0)

    # 音声データを書き出し
    write(output_path, sample_rate, audio_data)

    # スペクトトログラムをrfftで計算
    j = 0
    for i in range(0,len(audio_data),height*2):
        tmp_amp = np.abs(np.fft.rfft(audio_data[i:i+height*2]))
        spectrogram_data[:,j] = tmp_amp
        j += 1

    # pltで表示
    plt.imshow(spectrogram_data, cmap='gray', origin='lower')
    plt.axis('off')
    plt.savefig('output_spectrogram.png', bbox_inches='tight', pad_inches=0)
    plt.show()

# 画像から音声ファイルへの変換とスペクトログラムの作成の実行
image_to_wav('input.png', 'output_audio.wav', duration=10, sample_rate=16000)
