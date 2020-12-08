# Audio-Language-Classifier

DEMO the web app: https://language-ai.herokuapp.com/
<p align="center">
  <img src="https://i.ibb.co/80F30KW/audio-ai.png">
</p>

<p align="center">Built with Pytorch, Streamlit, and deployed on Heroku. Classifies 5 languages (EN, FR, GER, ES, IT) </p>

#### To Do List:
- Use Soundfile or pydub to enable both uploading mp3 and wav files (rather than just wav files currently)


**Overview of Project:**

The general idea of the project is to see if we can train a model to classify various languages. The process is to scrape wav files from a repository of user submitted voice clips (repository.voxforge1.org). The data consists of speakers with various accents, genders, and pitches for each language. 
Each voice clip (wav file) is to be preprocessed into melspectrograms with equal sample lengths (around a couple of seconds). Now, we can treat the melspectrograms as images and treat this as an image classification / recognition problem by running them through a Convolutional Neural Network model. I have seen a lot of people use this technique to classify music and instruments with melspectrograms, so I came up with the idea to try and see if I can classify audio languages with this approach. 

I decided to convert into melspectrograms because it converts audio waves by transforming them into frequency by time over amplitude. Then it converts the frequency scale into the mel-scale, which a non-linear transformation that is based on pitch comparisons. This makes the audio data much more easier to visualize and differentiate. 


**Files:**

1. `run_setup.sh` - bash script used to mass download files from voxforge of user submitted speech clips in multiple langauges. It then extracts and moves all .wav into separate subfolders for preprocessing automatically. Change the parameters in the script if you wish to download different languages. Default langauges are currently:  English, French, Spanish, German

2. `create_melspectrograms.py` - Python script to convert all the .wav files into melspectrograms and split them into train / test folders with each language as a class folder.

3. `model.py` - Final model architecture code using Pytorch. Convolutional Neural Network with batch norm and dropout. Found this to be the best performing model.

4. `helpers.py` - Helper utility functions such as evaluating, scoring, fitting, plot metrics, create confusion matrix for models. Plus GPU moving functions that were taken from FreeCodeCamp's pytorch tutorials.

5. `train_model.py` - Python script to train the model using the melspectrograms that were created. Use this AFTER running `create_melspectrograms.py`. Output will be a saved model state.

6. `notebooks/Prototyping_CNN_Models.ipynb` - created neural network models from scratch as class objects using Object Oriented Programming and Pytorch. They can be instantiated with hyperparameters. Created helper functions to fit, evaluate, and score the model. Prototyped several different models and tuned hyperparameters. Read the notebook for the whole process. 

7. `notebooks/Exploring Generalizability.ipynb` - scraped audio files from YouTube to test the model how it performs on real-world external data. Evaluating how generalizable the models are after only being trained using voice clips from one data repository. Then selecting the best performing model from both the prototyping and generalizability evaluations as our final exhaustive model to use for the web app. Read the notebook for the whole process.

8. `app.py` - Application created using Streamlit to allow users to upload their own voice clips and have the app return them the detected language. We are using the best performing model from the previous tests. To account for the lowered performance on external data, the app now extracts as many 4 second samples as it can from the audio file and classifies each one. It then aggregates the total count and returns a break down in percentage per language. This way we are looking at the holistic sample rather than only a 4 second window per file. This helps improve accuracy as we are sampling more times from a single file and returning more averaged results.

9. `assets` - folder holding assets (images, audio presets) for the app. 




