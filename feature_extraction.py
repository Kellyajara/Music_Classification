import numpy as np
import librosa

# Define your audio preprocessing function
def mfcc_features(file_path):
    # Loading audio file to librosa
    y, sr = librosa.load(file_path, offset=0, duration=6)
    
    # obtain mfcc features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512, n_fft=2048)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_var = mfcc.std(axis=1)
    mfcc_min = mfcc.min(axis=1)
    mfcc_max = mfcc.max(axis=1)
    
    mfcc_1 = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_fft=2048, n_mfcc=40)
    mfcc1_mean = mfcc_1.mean(axis=1)
    mfcc1_var = mfcc_1.std(axis=1)
    mfcc1_min = mfcc_1.min(axis=1)
    mfcc1_max = mfcc_1.max(axis=1)
    
    mfcc_2 = librosa.feature.mfcc(y=y, sr=sr, hop_length=256, n_fft=1024, n_mfcc=40)
    mfcc2_mean = mfcc_2.mean(axis=1)
    mfcc2_var = mfcc_2.std(axis=1)
    mfcc2_min = mfcc_2.min(axis=1)
    mfcc2_max = mfcc_2.max(axis=1)
    
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mean = delta_mfcc.mean(axis=1)
    delta_var = delta_mfcc.std(axis=1)
    
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta2_mean = delta2_mfcc.mean(axis=1)
    delta2_var = delta2_mfcc.std(axis=1)
    
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    mfcc_harmonic = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=40)
    harmonic_mean = mfcc_harmonic.mean(axis=1)
    harmonic_var = mfcc_harmonic.std(axis=1)
    
    mfcc_percussive = librosa.feature.mfcc(y=y_percussive, sr=sr, n_mfcc=40)
    percussive_mean = mfcc_percussive.mean(axis=1)
    percussive_var = mfcc_percussive.std(axis=1)

    
    mfcc_feature = np.concatenate((mfcc_mean, mfcc_var, mfcc_min, mfcc_max, 
                                  mfcc1_mean, mfcc1_var, mfcc1_min, mfcc1_max,
                                  mfcc2_mean, mfcc2_var, mfcc2_min, mfcc2_max,
                                  delta_mean, delta_var, delta2_mean, delta2_var,
                                  harmonic_mean, harmonic_var, 
                                  percussive_mean, percussive_var))

    return mfcc_feature

def extract_features(file_path):
    y, sr = librosa.load(file_path, offset = 0, duration = 6)
    #obtain melspectogram features
    melspec = librosa.feature.melspectrogram(y=y, sr=sr)
    melspec_mean = melspec.mean(axis=1)
    melspec_var = melspec.std(axis=1)
    melspec_min = melspec.min(axis=1)
    melspec_max = melspec.max(axis=1)

    # obtain spectral centroid features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = centroid.mean(axis=1)
    centroid_var = centroid.std(axis=1)
    centroid_min = centroid.min(axis=1)
    centroid_max = centroid.max(axis=1)

    # obtain chroma vector features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_var = chroma.std(axis =1)
    chroma_min = chroma.min(axis=1)
    chroma_max = chroma.max(axis=1)

    #getting tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tntz_mean = tonnetz.mean(axis=1)
    tntz_var = tonnetz.std(axis=1)
    tntz_min = tonnetz.min(axis=1)
    tntz_max = tonnetz.max(axis=1)

    #root-mean-squared
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean(axis = 1)
    rms_var = rms.std(axis = 1)

    #getting tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = np.array(librosa.feature.tempo(onset_envelope=onset_env, sr=sr))


    features = np.concatenate((melspec_mean, melspec_var, melspec_min, melspec_max,
                               centroid_mean, centroid_var, centroid_min, centroid_max,
                               chroma_mean, chroma_var, chroma_min, chroma_max,
                               tntz_mean, tntz_var, tntz_min, tntz_max,
                               rms_mean, rms_var, tempo))
    return features