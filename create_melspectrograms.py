import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np
from re import findall

# Create a function to convert wav files to images of melspectrograms in a given folder
def create_melspec(fold, train=8000, test=2000, n_fft = 2048, hop_length = 512, n_mels = 64, f_min = 20, f_max = 8000, sample_rate = 16000):
    '''
    Function to create mel-scaled spectrograms for all .wav files under a given language folder.
    Mel-scaled spectrograms (melspectrograms) are spectrograms with frequency converted to the mel-scale (a non-linear transformation)

    Inputs:
    fold = the wav files folder location RELATIVE to the currently working directory
    train = the number of training samples to generate
    test = the number of test samples to generate

    technical inputs:
    n_fft = 2048            # number of frames for the fast fornier transformation
    hop_length = 512        # window (hop length) of frames to analyse at a time
    n_mels = 64             # the number of mel filters to apply
    f_min = 20              # minimum frequency
    f_max = 8000            # max frequency
    sample_rate = 16000     # rate per second the time is audio is sampled. 16000 ~==

    Output:
    A spectrogram image for each .wav file under a folder: 'spectrograms' and respective subfolders for each language

    Requirements:
    - folder should only contain .wav files
    - if you used the bash script to download and extract the data, this should be automatically set up correctly

    Example:
    create_melspec(fold='en_wav', train=9000, test=1000)
        > returns: <train/test>/<language>/ <image of melspectrogram>.png ... for every wav file in 'en_wav' with a train/test split
    '''
    print(f'Creating melspectrograms in {fold}')

    # get list of files under the given folder
    wav_files = os.listdir(path=fold)

    # creates subdirectories for the output of melspectrogram images (train/test split)
    # NOTE: you will need to manually add more for additional languages you choose to include
    if 'en' in fold:
        spectrogram_path_train = "data/train/English"
        spectrogram_path_test = "data/test/English"
    elif 'fr' in fold:
        spectrogram_path_train = "data/train/French"
        spectrogram_path_test = "data/test/French"
    elif 'es' in fold:
        spectrogram_path_train = "data/train/Spanish"
        spectrogram_path_test = "data/test/Spanish"
    elif 'de' in fold:
        spectrogram_path_train = "data/train/German"
        spectrogram_path_test = "data/test/German"
    elif 'it' in fold:
        spectrogram_path_train = "data/train/Italian"
        spectrogram_path_test = "data/test/Italian"
    ### you can keep adding to this line for all languages you choose to include ###

    # create the path for the spectrograms if it doesn't exist already
    if os.path.isdir(spectrogram_path_train):
        pass
    else:
        os.makedirs(spectrogram_path_train)

    # do the same for test
    if os.path.isdir(spectrogram_path_test):
        pass
    else:
        os.makedirs(spectrogram_path_test)

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

        # decide where to store the files (train or test)
        if counter >= train:
            filename  = f'{spectrogram_path_test}/{new_name}'
        else:
            filename  = f'{spectrogram_path_train}/{new_name}'

        # save the output image
        plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
        plt.close('all')

        counter+=1

        # cap the max number of data for each class to have a balanced distribution
        if counter >= (train+test):
            print("Finished. Outputs are in the 'data' folder.")
            break

# return a list of all the language_wav folders in the working directory
all_folders = os.listdir(path="./")
r = re.compile(".*_wav")
lang_folders = list(filter(r.match, all_folders))

# check the list of folders
print(f"The wav files are stored in the folders: {lang_folders}")


# Generate spectrograms for each language folder
# NOTE: Since the lowest number of data points is italian with 10500 files, we will distribute the train/test based on that using amount with a 75/25 split
for i in lang_folders:
    create_melspec(fold=i, train=7875, test=2625)
