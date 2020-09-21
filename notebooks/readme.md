## Files

`Prototyping_CNN_Models.ipynb` 
- Created neural network models from scratch as class objects using Object Oriented Programming and Pytorch. They can be instantiated with hyperparameters. 
- Created helper functions to help evaluate, score, and fit the models
- Prototyped several different models
- Tuned hyperparameters such as epochs, learning rate, gradient clipping, and weight decay. 

`Exploring_Generalizability.ipynb` 
- Scraped audio files from YouTube to test the model how it performs on real-world external data.
- Converted those audio files into 4 second samples and then into melspectrograms
- Evaluated how generalizable the models are after only being trained using voice clips from one data repository. 
- Selected the best performing model from both the prototyping and generalizability evaluations as our final exhaustive model to use for the web app.
