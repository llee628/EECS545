from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import librosa
from pydub import AudioSegment
import soundfile as sf
from playsound import playsound
import os
import timeit

start = timeit.default_timer()

#%%
sound1 = AudioSegment.from_file('piano/piano.wav')
sound2 = AudioSegment.from_file('flute/flute.wav')
sound3 = AudioSegment.from_file('guitar/guitar.wav')
sound4 = AudioSegment.from_file('harp/harp.wav')
sound5 = AudioSegment.from_file('violin/violin.wav')
sound6 = AudioSegment.from_file('noise/noise.wav')

combined = sound1.overlay(sound2)
combined = combined.overlay(sound3)
combined = combined.overlay(sound4)
combined = combined.overlay(sound5)
combined = combined.overlay(sound6)
combined.export("combined/combined.wav", format='wav')



print('1/5')

piano_rate, p = wavfile.read('piano/piano.wav', 'r')
flute_rate, f = wavfile.read('flute/flute.wav', 'r')
guitar_rate, g = wavfile.read('guitar/guitar.wav', 'r')
harp_rate, h = wavfile.read('harp/harp.wav', 'r')
violin_rate, v = wavfile.read('violin/violin.wav', 'r')
noise_rate, n = wavfile.read('noise/noise.wav', 'r')


samplerate = piano_rate

p_1 = p * np.random.uniform(0.2, 1.0)
f_1 = f * np.random.uniform(0.2, 1.0)
g_1 = g * np.random.uniform(0.2, 1.0)
h_1 = h * np.random.uniform(0.2, 1.0)
v_1 = v * np.random.uniform(0.2, 1.0)
n_1 = n * np.random.uniform(0.1, 0.2)


wavfile.write("piano/piano1.wav", samplerate, p_1.astype(np.int16))
wavfile.write("flute/flute1.wav", samplerate, f_1.astype(np.int16))
wavfile.write("guitar/guitar1.wav", samplerate, g_1.astype(np.int16))
wavfile.write("harp/harp1.wav", samplerate, h_1.astype(np.int16))
wavfile.write('violin/violin1.wav', samplerate, v_1.astype(np.int16))
wavfile.write('noise/noise1.wav', samplerate, n_1.astype(np.int16))


#%%

sound1 = AudioSegment.from_file('piano/piano1.wav')
sound2 = AudioSegment.from_file('flute/flute1.wav')
sound3 = AudioSegment.from_file('guitar/guitar1.wav')
sound4 = AudioSegment.from_file('harp/harp1.wav')
sound5 = AudioSegment.from_file('violin/violin1.wav')
sound6 = AudioSegment.from_file('noise/noise1.wav')


combined = sound1.overlay(sound2)
combined = combined.overlay(sound3)
combined = combined.overlay(sound4)
combined = combined.overlay(sound5)
combined = combined.overlay(sound6)
combined.export("combined/combined1.wav", format='wav')


#%%
print('2/5')
p_1 = p * np.random.uniform(0.2, 1.0)
f_1 = f * np.random.uniform(0.2, 1.0)
g_1 = g * np.random.uniform(0.2, 1.0)
h_1 = h * np.random.uniform(0.2, 1.0)
v_1 = v * np.random.uniform(0.2, 1.0)
n_1 = n * np.random.uniform(0.1, 0.2)


wavfile.write("piano/piano2.wav", samplerate, p_1.astype(np.int16))
wavfile.write("flute/flute2.wav", samplerate, f_1.astype(np.int16))
wavfile.write("guitar/guitar2.wav", samplerate, g_1.astype(np.int16))
wavfile.write("harp/harp2.wav", samplerate, h_1.astype(np.int16))
wavfile.write('violin/violin2.wav', samplerate, v_1.astype(np.int16))
wavfile.write('noise/noise2.wav', samplerate, n_1.astype(np.int16))


sound1 = AudioSegment.from_file('piano/piano2.wav')
sound2 = AudioSegment.from_file('flute/flute2.wav')
sound3 = AudioSegment.from_file('guitar/guitar2.wav')
sound4 = AudioSegment.from_file('harp/harp2.wav')
sound5 = AudioSegment.from_file('violin/violin2.wav')
sound6 = AudioSegment.from_file('noise/noise2.wav')


combined = sound1.overlay(sound2)
combined = combined.overlay(sound3)
combined = combined.overlay(sound4)
combined = combined.overlay(sound5)
combined = combined.overlay(sound6)
combined.export("combined/combined2.wav", format='wav')


#%%
print('3/5')
p_1 = p * np.random.uniform(0.2, 1.0)
f_1 = f * np.random.uniform(0.2, 1.0)
g_1 = g * np.random.uniform(0.2, 1.0)
h_1 = h * np.random.uniform(0.2, 1.0)
v_1 = v * np.random.uniform(0.2, 1.0)
n_1 = n * np.random.uniform(0.1, 0.2)


wavfile.write("piano/piano3.wav", samplerate, p_1.astype(np.int16))
wavfile.write("flute/flute3.wav", samplerate, f_1.astype(np.int16))
wavfile.write("guitar/guitar3.wav", samplerate, g_1.astype(np.int16))
wavfile.write("harp/harp3.wav", samplerate, h_1.astype(np.int16))
wavfile.write('violin/violin3.wav', samplerate, v_1.astype(np.int16))
wavfile.write('noise/noise3.wav', samplerate, n_1.astype(np.int16))


sound1 = AudioSegment.from_file('piano/piano3.wav')
sound2 = AudioSegment.from_file('flute/flute3.wav')
sound3 = AudioSegment.from_file('guitar/guitar3.wav')
sound4 = AudioSegment.from_file('harp/harp3.wav')
sound5 = AudioSegment.from_file('violin/violin3.wav')
sound6 = AudioSegment.from_file('noise/noise3.wav')


combined = sound1.overlay(sound2)
combined = combined.overlay(sound3)
combined = combined.overlay(sound4)
combined = combined.overlay(sound5)
combined = combined.overlay(sound6)
combined.export("combined/combined3.wav", format='wav')


#%%
print('4/5')
p_1 = p * np.random.uniform(0.2, 1.0)
f_1 = f * np.random.uniform(0.2, 1.0)
g_1 = g * np.random.uniform(0.2, 1.0)
h_1 = h * np.random.uniform(0.2, 1.0)
v_1 = v * np.random.uniform(0.2, 1.0)
n_1 = n * np.random.uniform(0.1, 0.2)


wavfile.write("piano/piano4.wav", samplerate, p_1.astype(np.int16))
wavfile.write("flute/flute4.wav", samplerate, f_1.astype(np.int16))
wavfile.write("guitar/guitar4.wav", samplerate, g_1.astype(np.int16))
wavfile.write("harp/harp4.wav", samplerate, h_1.astype(np.int16))
wavfile.write('violin/violin4.wav', samplerate, v_1.astype(np.int16))
wavfile.write('noise/noise4.wav', samplerate, n_1.astype(np.int16))


sound1 = AudioSegment.from_file('piano/piano4.wav')
sound2 = AudioSegment.from_file('flute/flute4.wav')
sound3 = AudioSegment.from_file('guitar/guitar4.wav')
sound4 = AudioSegment.from_file('harp/harp4.wav')
sound5 = AudioSegment.from_file('violin/violin4.wav')
sound6 = AudioSegment.from_file('noise/noise4.wav')


combined = sound1.overlay(sound2)
combined = combined.overlay(sound3)
combined = combined.overlay(sound4)
combined = combined.overlay(sound5)
combined = combined.overlay(sound6)
combined.export("combined/combined4.wav", format='wav')


#%%
print('5/5')
p_1 = p * np.random.uniform(0.2, 1.0)
f_1 = f * np.random.uniform(0.2, 1.0)
g_1 = g * np.random.uniform(0.2, 1.0)
h_1 = h * np.random.uniform(0.2, 1.0)
v_1 = v * np.random.uniform(0.2, 1.0)
n_1 = n * np.random.uniform(0.1, 0.2)


wavfile.write("piano/piano5.wav", samplerate, p_1.astype(np.int16))
wavfile.write("flute/flute5.wav", samplerate, f_1.astype(np.int16))
wavfile.write("guitar/guitar5.wav", samplerate, g_1.astype(np.int16))
wavfile.write("harp/harp5.wav", samplerate, h_1.astype(np.int16))
wavfile.write('violin/violin5.wav', samplerate, v_1.astype(np.int16))
wavfile.write('noise/noise5.wav', samplerate, n_1.astype(np.int16))


sound1 = AudioSegment.from_file('piano/piano5.wav')
sound2 = AudioSegment.from_file('flute/flute5.wav')
sound3 = AudioSegment.from_file('guitar/guitar5.wav')
sound4 = AudioSegment.from_file('harp/harp5.wav')
sound5 = AudioSegment.from_file('violin/violin5.wav')
sound6 = AudioSegment.from_file('noise/noise5.wav')


combined = sound1.overlay(sound2)
combined = combined.overlay(sound3)
combined = combined.overlay(sound4)
combined = combined.overlay(sound5)
combined = combined.overlay(sound6)
combined.export("combined/combined5.wav", format='wav')


#%%

stop = timeit.default_timer()
print('Time: ', stop - start)

start = timeit.default_timer()
from sklearn.decomposition import FastICA

N = 44100*2800

_, w_1 = wavfile.read('combined/combined1.wav', 'r')
_, w_2 = wavfile.read('combined/combined2.wav', 'r')
_, w_3 = wavfile.read('combined/combined3.wav', 'r')
_, w_4 = wavfile.read('combined/combined4.wav', 'r')
_, w_5 = wavfile.read('combined/combined5.wav', 'r')



w_1 = w_1[:N, 0]
w_2 = w_2[:N, 0]
w_3 = w_3[:N, 0]
w_4 = w_4[:N, 0]
w_5 = w_5[:N, 0]


X = list(zip(w_1, w_2, w_3, w_4, w_5))
# plt.plot(np.arange(2000), X[:2000])
# Initialize FastICA with n_components=3
ica = FastICA(n_components=5)

stop = timeit.default_timer()
print('Time: ', stop - start)

start = timeit.default_timer()
# Run the FastICA algorithm using fit_transform on dataset X
ica_result = ica.fit_transform(X)
result_signal_1 = ica_result[:, 0]
result_signal_2 = ica_result[:, 1]
result_signal_3 = ica_result[:, 2]
result_signal_4 = ica_result[:, 3]
result_signal_5 = ica_result[:, 4]


# Convert to int, map the appropriate range, and increase the volume a little bit

big = 32767*100
result_signal_1_int = np.int16(result_signal_1*big)
result_signal_2_int = np.int16(result_signal_2*big)
result_signal_3_int = np.int16(result_signal_3*big)
result_signal_4_int = np.int16(result_signal_4*big)
result_signal_5_int = np.int16(result_signal_5*big)


wavfile.write("Result/result_signal_1.wav", samplerate, result_signal_1_int.astype(np.int16))
wavfile.write("Result/result_signal_2.wav", samplerate, result_signal_2_int.astype(np.int16))
wavfile.write("Result/result_signal_3.wav", samplerate, result_signal_3_int.astype(np.int16))
wavfile.write("Result/result_signal_4.wav", samplerate, result_signal_4_int.astype(np.int16))
wavfile.write("Result/result_signal_5.wav", samplerate, result_signal_5_int.astype(np.int16))


# Convert to int, map the appropriate range, and increase the volume a little bit
# result_signal_1_int = np.int16(result_signal_1*32767*100)
# result_signal_2_int = np.int16(result_signal_2*32767*100)
# result_signal_3_int = np.int16(result_signal_3*32767*100)

# plot_signal(result_signal_1_int)
# plot_signal(result_signal_2_int)
# plot_signal(result_signal_3_int)



def mix_musci(file_name, factor, export_name):
    for i in range(len(file_name)):
        samplerate, w = wavfile.read(f"{file_name[i]}.wav", 'r')
        w = w * factor[i]
        wavfile.write(f"{file_name[i]}_temp.wav", samplerate, w.astype(np.int16))
        sound = AudioSegment.from_file(f"{file_name[i]}_temp.wav")
        if i == 0:
            combined = sound
        else:
            combined = combined.overlay(sound)
    combined.export(f"{export_name}.wav", format='wav')


stop = timeit.default_timer()
print('Time: ', stop - start)