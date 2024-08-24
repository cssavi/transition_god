## This will become the best transition finder and will change how music is enjoyed

### 08/24/2024:
#### Feature Extraction:
The code extracts audio features (chroma, MFCC, and spectral contrast) from each song using librosa.
These features are extracted for the first few seconds of each song and stored in the cache.

#### Comparison:
When finding a match, it extracts the same features from the end of the query song.
It then compares these end features with the start features of each cached song.

#### Cosine Similarity:
The cosine similarity is calculated between the flattened feature vectors.
It measures the cosine of the angle between two vectors, indicating how similar their orientations are in the feature space.

## Future Plan

### Find a better way to capture the musical characteristics needed to determine a good transition
### Cache the first 10 & last 10 seconds to find the best song that transitions into and out of a song
### Create a seemingly infinite list of songs that auto-play
### Allow users to select the level of similarity that they are okay with and have it shuffle (no repeat)
