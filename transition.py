import librosa
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict
import os
import pickle

def extract_features(file_path, start=None, duration=1):
    # Load audio file
    y, sr = librosa.load(file_path, offset=start, duration=duration)
    
    # Extract features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Combine features
    features = np.vstack([chroma, mfcc, spectral_contrast])
    return features

class SongMatcher:
    def __init__(self, cache_file='song_features_cache.pkl', max_crossfade=10):
        self.start_features = defaultdict(dict)
        self.cache_file = cache_file
        self.max_crossfade = max_crossfade
        self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.start_features = pickle.load(f)
            print(f"Loaded {len(self.start_features)} cached song features.")

    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(dict(self.start_features), f)
        print(f"Saved {len(self.start_features)} song features to cache.")

    def add_start_song(self, song_id, song_path):
        if song_id not in self.start_features:
            # Extract features from the first max_crossfade seconds of start_song
            features = extract_features(file_path=song_path, start=0, duration=self.max_crossfade)
            self.start_features[song_id] = features
            self.save_cache()  # Save after adding each new song

    def load_songs_from_directory(self, directory_path):
        supported_extensions = ['.mp3', '.wav', '.flac', '.ogg']
        new_songs_added = 0
        
        for filename in os.listdir(directory_path):
            if any(filename.lower().endswith(ext) for ext in supported_extensions):
                song_path = os.path.join(directory_path, filename)
                song_id = os.path.splitext(filename)[0]  # Use filename without extension as song_id
                
                if song_id not in self.start_features:
                    self.add_start_song(song_id, song_path)
                    new_songs_added += 1
        
        print(f"Added {new_songs_added} new songs from the directory.")

    def find_best_match(self, end_song, crossfade_length):
        if crossfade_length > self.max_crossfade:
            raise ValueError(f"Crossfade length cannot exceed {self.max_crossfade} seconds.")

        # Get the duration of the end_song
        end_duration = librosa.get_duration(path=end_song)
        
        # Extract features from the last crossfade_length seconds of end_song
        end_features = extract_features(file_path=end_song, start=end_duration-crossfade_length, duration=crossfade_length)

        best_match = None
        best_similarity = -1

        for song_id, start_feature in self.start_features.items():
            # Calculate the number of frames to use based on the crossfade length
            num_frames = int(crossfade_length * start_feature.shape[1] / self.max_crossfade)
            
            # Use only the first crossfade_length seconds of the cached features
            start_feature_subset = start_feature[:, :num_frames]
            
            # Ensure both feature sets have the same number of frames
            min_frames = min(end_features.shape[1], start_feature_subset.shape[1])
            end_features_subset = end_features[:, :min_frames]
            start_feature_subset = start_feature_subset[:, :min_frames]
            
            # Flatten the features for comparison
            end_features_flat = end_features_subset.flatten()
            start_feature_subset_flat = start_feature_subset.flatten()
            
            similarity = 1 - cosine(end_features_flat, start_feature_subset_flat)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = song_id

        return best_match, best_similarity

# Example usage
matcher = SongMatcher(max_crossfade=10)

# Load songs from a directory
songs_directory = "songs"
matcher.load_songs_from_directory(songs_directory)

# Now you can perform queries with different crossfade lengths
end_song = "colors_of_the_wind.mp3"
for crossfade_length in [3, 5, 8, 10]:
    best_match, similarity = matcher.find_best_match(end_song, crossfade_length)
    print(f"Crossfade length: {crossfade_length}")
    print(f"Best matching song: {best_match}")
    print(f"Similarity score: {similarity}")
    print()