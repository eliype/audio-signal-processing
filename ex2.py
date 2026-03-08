
import librosa as lb
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.fft import fft, fftfreq

def task1(filepath,output_path):

    # loading sound with sample rate of 48000
    # it will allow us to add a sound with 23KHZ
    data, sr = lb.load(filepath,sr=48000)

    # the length of the sound
    duration = data.shape[0] / sr

    #samples across the time
    time = np.linspace(0, duration, len(data))

    # the watermark is a sin with 23KHZ
    # this sound cannot be heard by human so we will
    # add this frequency to the audio as good watermark
    watermark = 0.1*np.sin(2 * np.pi * 23000 * time)

    #adding watermark to data
    data_watermark = data + watermark

    # saving new file with good watermark
    sf.write(output_path+'/good_water_mark.wav', data_watermark, sr)

    # the watermark is a sin with 5KHZ
    # this sound can be heard by human so we will
    # add this frequency to the audio as bad watermark
    watermark = 0.1*np.sin(2 * np.pi * 5000 * time)

    # adding watermark to data
    data_watermark = data + watermark

    # saving new file with good watermark
    sf.write(output_path+'/bad_water_mark.wav', data_watermark, sr)

def task2():
    filename = "_watermarked.wav"
    group = dict()
    duration = 30
    for i in range(9):

        watermark = task2group("Task 2/"+str(i)+filename)
        if watermark in group.keys():
            group[watermark].append(str(i)+filename)
        else:
            group[watermark] = [str(i)+filename]


    for watermark,groups in group.items():

        print("frequency:",f"{watermark/duration:.5f}","- the audios", groups)


def task2group(filepath):

    # loading sound with sample rate of 48000
    data, sr = lb.load(filepath, sr=48000)

    # stft on the audio
    stft_data = lb.stft(data, n_fft=2048, hop_length=256, window='hann')

    # Display
    # stft_spectrogram = np.log(5000*np.abs(stft_data)+1)
    # plt.figure(figsize=(12, 6))
    # lb.display.specshow(stft_spectrogram, sr=sr, hop_length=256,
    #                     x_axis='time', y_axis='hz')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title("Spectrogram")
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Frequency (Hz)')
    # plt.show()

    return task2find_freq(stft_data)



def task2find_freq(stft_data):

    # for each time frame, find the amplitude at the peak frequency
    peak_amplitudes = []
    for i in range(stft_data.shape[1]):
        # look at high frequency region
        high_freq_region = stft_data[-200:, i]
        peak_amplitudes.append(np.max(high_freq_region))

    # display magnitude
    # plt.plot(peak_amplitudes)
    # plt.show()

    # apply FFT
    N = len(peak_amplitudes)
    yf = fft(peak_amplitudes)
    xf = fftfreq(N, 1)  # Assuming 1 sample spacing

    # find the dominant frequency (ignore DC component at 0)
    magnitude = np.abs(yf[:N // 2])
    dominant_freq_idx = np.argmax(magnitude[1:]) + 1
    dominant_frequency = xf[dominant_freq_idx]

    # number of waves = frequency × total length
    num_waves = abs(dominant_frequency) * N

    return num_waves

def task3(method1,method2):

    # loading sound with sample rate of 48000
    data1, sr1 = lb.load(method1, sr=12000)
    data2, sr2 = lb.load(method2, sr=48000)
    duration1 = data1.shape[0] / sr1
    # stft on the audio
    stft_data1 = lb.stft(data1, n_fft=2048, hop_length=256, window='hann')
    stft_data2 = lb.stft(data2, n_fft=2048, hop_length=256, window='hann')

    max_index1 = np.abs(stft_data1).argmax()
    max_freq_bin1 = max_index1 // stft_data1.shape[1]
    max_freq1 = (max_freq_bin1 * sr1)/2048

    max_index2 = np.abs(stft_data2).argmax()
    max_freq_bin2 = max_index2 // stft_data2.shape[1]
    max_freq2 = (max_freq_bin2 * sr2)/2048

    print("The slow factor:", max_freq2/max_freq1)

    # changing the values of the data in which the
    # differences between the lower values and the
    # bigger ones will be lesser. it will help to
    # visualize the data

    stft_spectrum1 = np.log(5000 * np.abs(stft_data1) + 1)
    stft_spectrum2 = np.log(5000 * np.abs(stft_data2) + 1)

    # Display
    plt.figure(figsize=(12, 6))
    lb.display.specshow(stft_spectrum1, sr=sr1, hop_length=256,
                        x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('method1')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')

    # Display
    plt.figure(figsize=(12, 6))
    lb.display.specshow(stft_spectrum2, sr=sr2, hop_length=256,
                        x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title("method2")
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

if __name__ == '__main__':
    #task1("Task 1/task1.wav","Task 1")
    task2()
    #task3("Task 3/task3_watermarked_method1.wav","Task 3/task3_watermarked_method2.wav")








