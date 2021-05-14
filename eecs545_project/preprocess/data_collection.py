from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


piano_rate, p_1 = wavfile.read('Result2_5comp/result_signal_1.wav', 'r')
flute_rate, f_1 = wavfile.read('Result2_5comp/result_signal_2.wav', 'r')
guitar_rate, g_1 = wavfile.read('Result2_5comp/result_signal_3.wav', 'r')
harp_rate, h_1 = wavfile.read('Result2_5comp/result_signal_4.wav', 'r')
violin_rate, v_1 = wavfile.read('Result2_5comp/result_signal_5.wav', 'r')

rate_list = [piano_rate, flute_rate, guitar_rate, harp_rate, violin_rate]
data_list = [p_1, f_1, g_1, h_1, v_1]

#for i in range(len(data_list)):
#    data_list[i] = data_list[i][abs(data_list[i]) > 0.1]

file_name_list = []
label_list = []
name_list = ['harp', 'violin', 'flute', 'guitar', 'piano']
index = 0

for rate, data in zip(rate_list, data_list):
    number = np.arange(100000)
    np.random.shuffle(number)
    st = 0
    i = 0
    print(index)
    while st < data.shape[0]:
        file_name = f"Result2_5comp/ICA_dataset/{name_list[index]}_piece{number[i]}.wav"
        wavfile.write(file_name, rate,
                    data[st:st+2*rate].astype(np.int16))
        file_name_list.append(f"{name_list[index]}_piece{number[i]}.wav")
        label_list.append(name_list[index])
        i += 1
        st += 2*rate
    index += 1


df = pd.DataFrame({'File_name':file_name_list, 'Label': label_list})
df = df.sample(frac=1)
df.to_excel('ICA_label_file.xlsx')
