# speech_identification_recognition
Train Gaussian Mixture Models for speech recognition and speaker identification

This project has two stages: the first trains mixtures of Gaussians to the acoustic characteristics of individual speakers and then identifies speaked based on these models. The second evaluates two speech recognition engines.

The data is CSC Deceptive Speech Corpus consisting of 32 hours of audio interview from 32 speakers of english.

First stage:

gmm.py - Implements 3 utility functions: log observation probability of x_t for the m_th mixture component, the log probability of m given x_t using model theta, the log likelihood of a set of data X. Then trains M-component GMM for each speaker in the data set. Finally, tests if the actual speaker is also the most likely speaker.

Second stage:

levenshtein.py - Evaluates two Automatic Speech Recognition engines: Kaldi and Google Speech API using the Word Error Rate derived from levenshtein distance.

Request the data.
