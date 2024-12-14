# Speaker Diarization

This repository contains a Python script for speaker diarization using `pyannote.audio`. The script extracts speaker embeddings from an audio file and groups segments by speaker ID.

## Setup Instructions

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/speaker-diarization.git
    cd speaker-diarization
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your Hugging Face token by creating a `.env` file in the root directory:
    ```txt
    HF_TOKEN=your_hugging_face_token_here
    ```

## Usage

1. Prepare your input audio file in WAV format and place it in the directory.
   
2. Run the diarization script:
    ```bash
    python diarization_pipeline.py
    ```

3. The script will generate a `diarized_output.json` file with speaker segments.

## Output Format

The output JSON file will contain the following structure:
```json
[
  {
    "start": 0.0,
    "end": 5.0,
    "speaker_id": 0
  },
  {
    "start": 5.0,
    "end": 10.0,
    "speaker_id": 1
  }
]
