import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
import torchvision
import torchvision.transforms as tf
import io
from PIL import Image
from collections import Counter

from model import CNN_model_3

############ SETUP ENVIRONMENT ################################################
st.set_option('deprecation.showfileUploaderEncoding', False)
st.beta_set_page_config(page_title='Language AI', initial_sidebar_state='expanded')

# Our labels for the classes (DO NOT CHANGE ORDER)
classes = ["English", "French", "German", "Italian", "Spanish"]

# transformations for our spectrograms
transformer = tf.Compose([tf.Resize([64,64]), tf.ToTensor()])

# load our saved model function with caching
@st.cache(allow_output_mutation=True)
def load_model(path="trained_model_3_state.pt"):
    model = CNN_model_3(opt_fun=torch.optim.Adam, lr=0.001)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_audio(file, sr=16000):
    clip, sample_rate = librosa.load(file, sr=sr)
    return clip, sample_rate

# load the model
model = load_model("assets/trained_model_3_state.pt")

"""
# Language Detection AI

AI bot trained to detect the language from speech using Deep Learning

##
"""
# landing screen
imagewidth = 550
placeholder = st.empty()
placeholder.image('assets/speechbubble.png', width=imagewidth)


############ SIDEBAR CONFIGURE ################################################

# Ask user to upload a voice file for language classification
st.sidebar.markdown('## Upload Voice Clip')
wav_file = st.sidebar.file_uploader("Upload Your WAV File Here:", type=["wav"])

# allow users to choose a preloaded sample
st.sidebar.markdown("""## Or Use a Preset Audio Clip:""")
preset = st.sidebar.radio("Choose a Language", options=["None"]+classes)

st.sidebar.header("") # add some empty space
st.sidebar.header("") # add some empty space

st.sidebar.markdown(
"""
-----------
## Instructions
1. Upload a voice clip or select a **preset sample** audio file above
2. Click on 'Start' to begin detecting the language
3. View the results
4. Upload a new file or choose another preset to try again
5. If you are getting 'NaN', use clips longer than 4 seconds
""")

st.sidebar.header("") # add some empty space
st.sidebar.header("") # add some empty space
st.sidebar.header("") # add some empty space

st.sidebar.markdown(
"""
-----------

## Github
Creator: Bowen Zhang

Project Repository: [Link Here](https://github.com/b-zhang93/Audio-Language-Classifier)
""")


############ RUN MODEL AND RETURN OUTPUT########################################
# if no files are uploaded, use preset ones by default
if wav_file is None:
    if preset == 'English':
        wav_file = 'assets/english_preset.wav'
    elif preset == "French":
        wav_file = 'assets/french_preset.wav'
    elif preset == "German":
        wav_file = 'assets/german_preset.wav'
    elif preset == "Italian":
        wav_file = 'assets/italian_preset.wav'
    elif preset == "Spanish":
        wav_file = 'assets/spanish_preset.wav'
    else:
        pass

if wav_file is not None:

    placeholder.image("assets/speechbubble2.png", width=imagewidth)
    st.audio(wav_file) # allows users to play back uploaded files

    # set up the progress animations and initialize empty lists
    status_text = st.empty()
    progress_bar = st.empty()
    predictions = []

    status_text.text('Press Start To Begin Detection...')
    if st.button('Start'):

        # load our audio file into array
        status_text.text('Rendering Audio File...')
        clip, sample_rate = load_audio(wav_file, sr=16000)

        duration = len(clip)
        num_samples = int(duration/60000) # number of samples we can extract from this file
        start = 0     # starting sample window
        end = 60000   # end sample window

        # take a sample from our uploaded voice clip
        for i in range(num_samples):
            prog = int(((i+1)/num_samples)*100)
            status_text.text(f"Analysing Audio: {prog}%")
            progress_bar.progress(prog)

            clip_new = clip[start:end]

            # initialize our plot for the melspectrogram
            fig = plt.figure(figsize=[0.75,0.75])
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            mel_spec = librosa.feature.melspectrogram(clip_new, n_fft=2048, hop_length=512, n_mels=64,
                                              sr=sample_rate, power=1.0, fmin=20, fmax=8000)
            librosa.display.specshow(librosa.amplitude_to_db(mel_spec, ref=np.max), fmax=8000, sr=sample_rate)

            mel = io.BytesIO()
            plt.savefig(mel, dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close('all')

            # load image to tensor and correct dimensions for our model
            image = Image.open(mel).convert('RGB')
            image = transformer(image).float()
            image = torch.tensor(image, requires_grad=True)
            image = image.unsqueeze(0)

            # run predictions
            model.eval()
            output = model(image)
            _, predicted = torch.max(output, dim=1)

            # record predictions and update sample windows
            predictions.append(classes[predicted[0].item()])
            start += 60000
            end += 60000

        # output our results now
        status_text.empty()
        progress_bar.empty()

        # tally up the predictions for each sample
        results = Counter(predictions)

        # placeholder value of 0 for languages that did not appear
        for c in classes:
            if c in results.keys():
                pass
            else:
                results[c] = 0

        # create a dataframe to show our results in percentage
        df = pd.DataFrame.from_dict(results, orient='index', columns=['percent']).T.div(num_samples)*100

        # get the max result (best prediction)
        highest_prediction = df.idxmax(axis=1)[0]
        if highest_prediction == 'English':
            placeholder.image('assets/speechbubbleENG.png', width=imagewidth)
        elif highest_prediction == "French":
            placeholder.image('assets/speechbubbleFR.png', width=imagewidth)
        elif highest_prediction == "German":
            placeholder.image('assets/speechbubbleGER.png', width=imagewidth)
        elif highest_prediction == "Spanish":
            placeholder.image('assets/speechbubbleES.png', width=imagewidth)
        else:
            placeholder.image('assets/speechbubbleIT.png', width=imagewidth)

        st.write(
        """
        ----------
        # Breakdown
        By percentage of languages the AI thinks it is
        """)
        # return the breakdown of how the model classified each sample
        st.dataframe(df, width=600)
        st.bar_chart(df.T, height=400, use_container_width=True)
