import streamlit as st
import numpy as np
import joblib
import librosa
from feature_extraction import mfcc_features, extract_features

# Load the music classification model
model = joblib.load('/Users/kellyjara/Desktop/Music_Classification/Data/knn_model.pkl')

def main():
    st.title('Music Genre Classification App')

    audio_file = st.file_uploader('Upload an audio file', type=['mp3', 'wav'])

    if audio_file:
        st.audio(audio_file, format='audio/ogg') 

        # Preprocess the audio file
        features_result = extract_features(audio_file)
        mfcc_features_result = mfcc_features(audio_file)

        mfcc_features_result = np.expand_dims(mfcc_features_result, axis =1)
        mfcc_features_result = mfcc_features_result[:features_result.shape[0]]
        
        print("Features Result Shape:", features_result.shape)
        print("MFCC Features Result Shape:", mfcc_features_result.shape)

        features_result = np.expand_dims(features_result, axis = 1)
        
        #concatenating features from both functions
        combined_features = np.hstack((features_result, mfcc_features_result))

        print("Combined Features Shape:", combined_features.shape)

        combined_features_2d = combined_features.reshape(1, -1)
        # Perform inference using the loaded model
        prediction = model.predict(combined_features_2d)

        # Get the predicted genre (or any other classification result)
        genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        predicted_genre = genres[prediction[0]]

        st.write(f'Predicted Genre: {predicted_genre}')
if __name__ == '__main__':
    main()



