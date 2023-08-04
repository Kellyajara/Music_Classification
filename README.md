# Music Genre Classification
![Screen Shot 2023-08-03 at 9 23 56 AM](https://github.com/Kellyajara/Music_Classification/assets/127794801/e94a1442-7195-4f27-a268-d2bf879140f4)

## Overview/Business Probelm
Streaming services have been around since the late 90's early 2000s. From services like Pandora to the now most popular Apple Music and Spotify, they all have something in common, playlist recommendations. At the base of those recommendations, how do these platforms know what to recommend to their users? This is where music classification models come in. Music classification takes information from various audio files and analyzes their characteristics to classify them as a specific genre. This form of modeling is helpful when building recommendation models as it is the baseline for recognizing patterns within audio samples. 


## Data
Data was obtained from audio samples, in .wav format , 1,000 of which were obtained from the GTZAN data set and the other 400 from my own personal music library. The audio files consisted of 10 different genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae and rock). Originally the audio samples were in 30 second increments but to increase the size of the data, I used a function to trim each audio down to 6 seconds. This allowed my data to go from 1400~ samples to about 6,987~. Once I had the samples I needed, I then created another set of functions that extracted the needed informatin from the audio samples. 

![genre_counts](https://github.com/Kellyajara/Music_Classification/assets/127794801/40d4d68f-e890-446e-a0c8-11446def9b6f)

The features extracted were: 
  * MFCC (Mel Frequency Cepstral Coeffeciants)
  * Mel Spectrogram
  * Harmonic/Percussive Waveform
  * Spectral Centroid
  * Chroma Vector
  * Tonnetz
  * Root Mean Squared
  * Tempo

![feature_importance](https://github.com/Kellyajara/Music_Classification/assets/127794801/6d9d28f1-0ac8-43c1-80a3-76f9fceab62f)


For the majority of my models, I extracted the mean, standard deviation, min and max values of the over-all audio samples, instead of extracting the values frame by frame as that would cause high-dimensionality. When working with my recurrent neural network models I opted for a different approach and took the values frame by frame from the MFCC and Mel Spectrogram features. This caused the dataframe to have high dimensionality but I was able to get the features down to 65. Once I had all of my features extracted, I then one-hot-encoded the genres column to create a multi-labels for this classification model.

## Modeling
  A Random Forest Model was the first model conducted with no hyperparameters. This model had an accuracy rate of 58, a precision rate of 98, and a recall rate of 58. Looking at the confusion matrix, I was able to see that the model was having issues classifying other genres as blues. From there, I knew that the majority of music classification models performed better with K-nearest neighbor models, and this is because KNN models base their predictions on the data points that are closest to each other. So if a genre has specific characteristics within the dataset, a knn model would have an easier time depicting that and creating the needed boundaries within that set. The KNN model had an accuracy, recall, and precision rate of 89. This model overall had a better ability of predicting the genre classes with a small percentage of them being misclassified. 
![KNN](https://github.com/Kellyajara/Music_Classification/assets/127794801/cd06db8d-383d-4c21-bcc8-30b9d894caab)
![confusion_matrix](https://github.com/Kellyajara/Music_Classification/assets/127794801/faa0270c-43c2-46bb-99df-a6542b466395)
  
  I decided to dive into deeper modeling and began to look into recurrent neural network models with long short-term memory. Because RNN models learn information sequentially, and music is written in a sequential pattern, this type of modeling would have a better chance to classify the different genres as it would learn what patterns are recurring in each cell of the neural network. RNNs have hidden layers that capture information from one output and use that information in the next sequence of the model and by using LSTM within the model, it is able to hold information for longer periods of time. I ran the RNN model with both MFCC and Mel spectrogram features. For the model with MFCC features, this model had about a 19% accuracy rate while the one with Mel Spectrogram features had about 32%. I believe this is because the mel spectrogram features tend to have more details on sound patterns than the MFFC.   

![blues_melspec](https://github.com/Kellyajara/Music_Classification/assets/127794801/924d6aca-0fa4-4124-9175-42fc6bdac929) ![blues_mfcc](https://github.com/Kellyajara/Music_Classification/assets/127794801/9f18e7e0-c614-439e-91cf-d2452358ec96)

![MFCC Accuracy](https://github.com/Kellyajara/Music_Classification/assets/127794801/04198fb5-19e8-4ff7-8eb5-3b7f38c21c00)
![MelSpec Accuracy](https://github.com/Kellyajara/Music_Classification/assets/127794801/732d315f-560a-4127-b723-e0bae582cb38)

  Because I did not have enough samples to create sufficient data, and then went and trimmed the audio down further to create more, I took away a large portion of what can help models, especially RNN models, distinguish what is so different about these various genres. For example, many people believe blues and jazz to be similar but there are certain sequences within the song/piece of music that distinguish the two genres from each other. The same goes for genres such as rock, metal, and alternative music. Features like MFCC and Mel Spectrogram are constructed using bands that are based on the Mel scale. The mel scale is a scale of pitches that represents the way the human auditory system responds to sound. 

## Conclusion & Further Analysis
Overall, the KNN model was able to accuractly predict the genres of various different audio samples 89% of the time with averages of specifc features. To further improve this model and possibly achieve a better outcome with neural networks I would need to look into collecting more audio samples of the various genres and keep the length of those audio files at a minimum of 30 seconds. Especially when working with neural networks it is importat that there is a sufficient amount of data for these models to learn any patterns and distinguished details through the different labels within the data set. 

Nest Steps:
  * Deploy music classification app that allows users to input an audio file and receive an output of that audio genre to show accuracy of the model.
  * Explore classifying more genres:
  * Explore music from different cultures and how it affects the performance of the model
  * Create a genre-based recommendation system for user playlists. 


Sources: 
Librosa - McFee, Brian, Matt McVicar, Daniel Faronbi, Iran Roman, Matan Gover, Stefan Balke, Scott Seyfarth, Ayoub Malek, Colin Raffel, Vincent Lostanlen, Benjamin van Niekirk, Dana Lee, Frank Cwitkowitz, Frank Zalkow, Oriol Nieto, Dan Ellis, Jack Mason, Kyungyun Lee, Bea Steers, … Waldir Pimenta. (2023). librosa/librosa: 0.10.0.post2 (0.10.0.post2). Zenodo. https://doi.org/10.5281/zenodo.7746972
GTZAN Dataset - https://datasets.activeloop.ai/docs/ml/datasets/gtzan-genre-dataset/

