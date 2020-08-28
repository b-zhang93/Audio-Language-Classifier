# Audio-Language-Classifier
Using deep learning to classify speech into multiple different languages. This project is a work in progress.

**Overview of Project:**

The general idea is to scrape wav files from a repository of user submitted voice clips (repository.voxforge1.org). The data consists of speakers with various accents, genders, and pitches for each language. 
Each voice clip (wav file) is to be preprocessed into melspectrograms with equal sample lengths (around a couple of seconds). Now, we can treat the melspectrograms as images and treat this as an image classification / recognition problem by running them through a Convolutional Neural Network model. I have seen a lot of people use this technique to classify music and instruments with melspectrograms, so I came up with the idea to try and see if I can classify audio languages with this approach. 

I decided to convert into melspectrogram because it converts audio waves by transforming them into frequency by time over amplitude. Then it converts the frequency scale into the mel-scale, which a non-linear transformation that is based on pitch comparisons. 


**Files:**

1. `run_setup.sh` - bash script used to mass download files from voxforge of user submitted speech clips in multiple langauges. It then extracts and moves all .wav into separate subfolders for preprocessing automatically. Change the parameters in the script if you wish to download different languages. Default langauges are currently:  English, French, Spanish, German
2. `create_melspectrograms.py` - Python script to convert all the .wav files into melspectrograms and split them into train / test folders with each language as a class folder.
3. `mel_transform.ipynb` - Juypter notebook used for prototyping the create_melspectrogram.py file. Will probably remove after final version since it's redundant.
4. `CNN.ipynb`(outdated) - prototyping first two simple CNN models with the melspectrograms. Will be updating later to use OOP and class / methods for the models and adding more complex models models to compare results. Code is very rough and messy.
5. `CNN_with_oop.ipynb` - converted the models and functions in `CNN.ipynb` into cleaner versions using Object Oriented Programming. Converted the models into class objects which can be instantiated with hyperparameters. Cleaned up the functions and minimalized everything. Will be using this notebook going forward.
