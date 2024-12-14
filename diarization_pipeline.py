import json
import os
import time
import logging
import librosa
import numpy as np
import warnings
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment
from pyannote.audio import Audio
from sklearn.cluster import SpectralClustering, KMeans
from dotenv import load_dotenv

# Suppress specific warnings related to PyTorch std() computation
warnings.filterwarnings("ignore", category=UserWarning, message=".*std\(\): degrees of freedom is <= 0.*")

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from the .env file
load_dotenv()

# Get the Hugging Face token from the environment
hf_token = os.getenv("HF_TOKEN")

# Load the speaker diarization pipeline with the authentication token
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)


def diarize_audio(input_audio):
    """
    Perform speaker diarization on the given audio file.

    Args:
    - input_audio: Path to the input audio file (WAV format).

    Returns:
    - segments: List of dictionaries with timestamps and corresponding speaker ID.
    """
    start_time = time.time()

    logging.info("Starting diarization pipeline...")
    with ProgressHook() as hook:
        diarization = pipeline(input_audio, hook=hook)

    diarization_time = time.time() - start_time
    logging.info(f"Diarization completed in {diarization_time:.2f} seconds.")

    segments = []
    for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
        segment = {
            "start": speech_turn.start,
            "end": speech_turn.end,
            "speaker_id": speaker
        }
        segments.append(segment)

    return segments, diarization_time


def cluster_speakers_spectral(segments, fixed_embedding_size=40000):
    """
    Clusters speaker segments using Spectral Clustering.
    """
    embeddings = []
    timestamps = []
    audio = Audio()

    # Use librosa to get the duration of the audio
    y, sr = librosa.load(input_audio, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)

    for segment in segments:
        excerpt = Segment(start=segment["start"], end=segment["end"])

        # Ensure the segment doesn't exceed the audio duration
        if excerpt.end > total_duration:
            excerpt = Segment(start=excerpt.start, end=total_duration)  # Clip to valid range

        waveform, sample_rate = audio.crop(input_audio, excerpt)

        embedding = waveform.flatten()
        if len(embedding) > fixed_embedding_size:
            embedding = embedding[:fixed_embedding_size]
        elif len(embedding) < fixed_embedding_size:
            embedding = np.concatenate([embedding, np.zeros(fixed_embedding_size - len(embedding))])

        embeddings.append(embedding)
        timestamps.append((segment["start"], segment["end"]))

    embeddings = np.vstack(embeddings)
    spectral = SpectralClustering(n_clusters=2, affinity='rbf', random_state=42)
    speaker_ids = spectral.fit_predict(embeddings)

    clustered_segments = []
    for i, (start, end) in enumerate(timestamps):
        clustered_segments.append({
            "start": start,
            "end": end,
            "speaker_id": int(speaker_ids[i])
        })

    return clustered_segments


def cluster_speakers_kmeans(segments, fixed_embedding_size=40000):
    """
    Clusters speaker segments using KMeans clustering.
    """
    embeddings = []
    timestamps = []
    audio = Audio()

    # Use librosa to get the duration of the audio
    y, sr = librosa.load(input_audio, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)

    for segment in segments:
        excerpt = Segment(start=segment["start"], end=segment["end"])

        # Ensure the segment doesn't exceed the audio duration
        if excerpt.end > total_duration:
            excerpt = Segment(start=excerpt.start, end=total_duration)  # Clip to valid range

        waveform, sample_rate = audio.crop(input_audio, excerpt)

        embedding = waveform.flatten()
        if len(embedding) > fixed_embedding_size:
            embedding = embedding[:fixed_embedding_size]
        elif len(embedding) < fixed_embedding_size:
            embedding = np.concatenate([embedding, np.zeros(fixed_embedding_size - len(embedding))])

        embeddings.append(embedding)
        timestamps.append((segment["start"], segment["end"]))

    embeddings = np.vstack(embeddings)
    kmeans = KMeans(n_clusters=2, random_state=42)
    speaker_ids = kmeans.fit_predict(embeddings)

    clustered_segments = []
    for i, (start, end) in enumerate(timestamps):
        clustered_segments.append({
            "start": start,
            "end": end,
            "speaker_id": int(speaker_ids[i])
        })

    return clustered_segments


def load_json(file_path):
    """
    Load a JSON file containing diarized segments.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_der(spectral_file, kmeans_file):
    """
    Calculate the Diarization Error Rate (DER) by comparing Spectral Clustering and KMeans clustering results.

    Args:
    - spectral_file: Path to the Spectral Clustering results JSON file.
    - kmeans_file: Path to the KMeans Clustering results JSON file.

    Returns:
    - DER: The Diarization Error Rate as a float value.
    """
    # Load the diarized segments for both Spectral and KMeans
    spectral_segments = load_json(spectral_file)
    kmeans_segments = load_json(kmeans_file)

    # Ensure the same number of segments in both
    if len(spectral_segments) != len(kmeans_segments):
        logging.warning("Mismatch in the number of segments between Spectral Clustering and KMeans. Check the input files.")
        return 0

    correct_count = 0
    total_segments = len(spectral_segments)
    missed_speech = 0
    false_alarm = 0

    for spectral, kmeans in zip(spectral_segments, kmeans_segments):
        spectral_start, spectral_end, spectral_id = spectral['start'], spectral['end'], spectral['speaker_id']
        kmeans_start, kmeans_end, kmeans_id = kmeans['start'], kmeans['end'], kmeans['speaker_id']

        # Check if the segments are the same (within a tolerance)
        if spectral_start == kmeans_start and spectral_end == kmeans_end:
            # Compare speaker_id
            if spectral_id == kmeans_id:
                correct_count += 1
            else:
                missed_speech += (spectral_end - spectral_start)  # Missed speech when speaker_id doesn't match
                false_alarm += (kmeans_end - kmeans_start)  # False alarm when speaker_id doesn't match

    # Calculate accuracy
    accuracy = correct_count / total_segments if total_segments > 0 else 0

    # Calculate DER (Diarization Error Rate)
    der = (missed_speech + false_alarm) / total_segments if total_segments > 0 else 0

    return accuracy, der


def save_to_json(segments, output_file):
    """
    Save diarized segments with speaker IDs into a JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(segments, f, indent=4)

    logging.info(f"Results saved to {output_file}")


def main(input_audio, output_json_spectral, output_json_kmeans):
    """
    Main function to run the diarization pipeline, save the results, and calculate metrics.
    """
    # Step 1: Perform diarization
    segments, diarization_time = diarize_audio(input_audio)

    # Step 2: Cluster speaker segments using Spectral Clustering
    start_clustering_time = time.time()
    clustered_segments_spectral = cluster_speakers_spectral(segments)
    clustering_time = time.time() - start_clustering_time
    logging.info(f"Spectral Clustering completed in {clustering_time:.2f} seconds.")

    # Save Spectral Clustering results
    save_to_json(clustered_segments_spectral, output_json_spectral)

    # Step 3: Cluster speaker segments using KMeans Clustering
    start_clustering_time = time.time()
    clustered_segments_kmeans = cluster_speakers_kmeans(segments)
    clustering_time = time.time() - start_clustering_time
    logging.info(f"KMeans Clustering completed in {clustering_time:.2f} seconds.")

    # Save KMeans Clustering results
    save_to_json(clustered_segments_kmeans, output_json_kmeans)

    # Step 4: Calculate DER and Accuracy
    accuracy, der = calculate_der(output_json_spectral, output_json_kmeans)
    logging.info(f"Accuracy: {accuracy:.2f}")
    logging.info(f"Diarization Error Rate (DER): {der:.2f}")

    # Step 5: Log total runtime
    total_runtime = diarization_time + clustering_time
    logging.info(f"Total runtime: {total_runtime:.2f} seconds.")
    if total_runtime > 0:
        real_time_factor = total_runtime / (len(input_audio) / 1000.0)  # Assume 1 second per sample
        logging.info(f"Real-time feasibility factor: {real_time_factor:.2f}")


if __name__ == "__main__":
    input_audio = 'input_audio.wav'  # Example input audio file
    output_json_spectral = 'diarized_spectral_output.json'
    output_json_kmeans = 'diarized_kmeans_output.json'

    # Run the diarization process for both clustering methods
    main(input_audio, output_json_spectral, output_json_kmeans)
