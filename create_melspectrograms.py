import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np
from re import findall

# set up the parameters for conversion to mel spectrogram (should I include these into the function or class I am going to create to generate spectrograms?)
n_fft = 2048
hop_length = 512
n_mels = 64
f_min = 20
f_max = 8000
sample_rate = 16000

# Create a function to convert wav files to images of melspectrograms in a given folder
def create_melspec(fold, MAX=10000):
    '''
    Function to create spectrograms for all .wav files under a gven language folder

    Inputs:
    fold = the wav files folder location RELATIVE to the currently working directory
    MAX = the maximum number of melspectrograms / data points to generate

    Output:
    A spectrogram image for each .wav file under a folder: 'spectrograms' and respective subfolders for each language

    Requirements:
    - folder should only contain .wav files
    - if you used the bash script to download and extract the data, this should be automatically set up correctly

    Example:
    create_melspec(fold='en_wav', MAX=7000)
        > returns: spectrograms/en_data/ <image of melspectrogram>.png ... for every wav file in 'en_wav'
    '''

    print(f'Processing spectrograms in {fold}')

    # get list of files under the given folder
    wav_files = os.listdir(path=fold)

    # creates subdirectories for the output of melspectrogram images
    spectrogram_path = f'spectrograms/{fold.replace("_wav", "_data")}'
    if os.path.isdir(spectrogram_path):
        pass
    else:
        os.makedirs(spectrogram_path)

    # convert wav to melspectrograms and saves the image
    counter = 0
    for audio_file in wav_files:
        clip, sample_rate = librosa.load(path=f'{fold}/{audio_file}')
        duration = len(clip)

        if duration >= 76000:
            clip = clip[16000:16000+60000]
        else:
            clip = clip[:60000]

        # initialize our plot for the melspectrogram
        fig = plt.figure(figsize=[0.75,0.75])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)

        # create the melspectrogram as a plot
        mel_spec = librosa.feature.melspectrogram(clip, n_fft=n_fft, hop_length=hop_length, n_mels = n_mels,
                                          sr=sample_rate, power=1.0, fmin=f_min, fmax=f_max)
        librosa.display.specshow(librosa.amplitude_to_db(mel_spec, ref=np.max), fmax=f_max, sr=sample_rate)

        # extract the speaker from filename and rename with autoincrement key
        speaker = findall(r'(?<=wav_).*?[._-]', audio_file)
        new_name = audio_file.replace(audio_file, f'{speaker[0]}{counter}.png')
        filename  = f'{spectrogram_path}/{new_name}'

        # save the output image
        plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
        plt.close('all')

        counter+=1

        # cap the max number of data for each class to have a balanced distribution
        if counter >= MAX:
            break

# create our spectrograms
lang_folders = ['en_wav','fr_wav','es_wav','de_wav']

for i in lang_folders:
    create_melspec(i, MAX=6500)
