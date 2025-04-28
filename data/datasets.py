"""
Dataset implementations for ACMFT emotion recognition.

This module provides dataset classes for loading and processing multimodal
emotional data from standard datasets like IEMOCAP and RAVDESS.
"""

import glob
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.preprocessing.audio import preprocess_audio
from data.preprocessing.hr import preprocess_hr_signal
from data.preprocessing.visual import preprocess_face_image


class EmotionDataset(Dataset):
    """
    Base class for multimodal emotion recognition datasets.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        modalities: List[str] = ["visual", "audio", "hr"],
        transform: Optional[Dict[str, Callable]] = None,
        preprocess: bool = True,
        cache_processed: bool = True,
        segment_length: float = 3.0,  # in seconds
        emotion_map: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize dataset.

        Args:
            root_dir: Root directory of the dataset
            split: Data split ("train", "val", "test")
            modalities: List of modalities to load
            transform: Dictionary of transform functions for each modality
            preprocess: Whether to preprocess data
            cache_processed: Whether to cache preprocessed data
            segment_length: Length of each segment in seconds
            emotion_map: Mapping from emotion names to indices
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.modalities = modalities
        self.transform = transform or {}
        self.preprocess = preprocess
        self.cache_processed = cache_processed
        self.segment_length = segment_length

        # Default emotion mapping if not provided
        self.emotion_map = emotion_map or {
            "neutral": 0,
            "happiness": 1,
            "sadness": 2,
            "anger": 3,
            "fear": 4,
            "surprise": 5,
            "disgust": 6,
            "contempt": 7,
        }

        # Initialize samples list
        self.samples = []

        # Cache directory
        self.cache_dir = self.root_dir / "cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Clear stale cache if caching is disabled
        if not self.cache_processed:
            cache_file = self.cache_dir / f"{self.split}_cache.pkl"
            if cache_file.exists():
                cache_file.unlink()
            # Initialize empty cache
            self.cache = {}

        # Load dataset
        self._load_dataset()

        # Load or create cache
        if self.cache_processed:
            self._setup_cache()

    def _load_dataset(self):
        """
        Load dataset samples.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _load_dataset")

    def _setup_cache(self):
        """
        Set up cache for preprocessed data.
        """
        cache_file = self.cache_dir / f"{self.split}_cache.pkl"

        if cache_file.exists():
            # Load cached data
            logging.info(f"Loading cached data from {cache_file}")
            with open(cache_file, "rb") as f:
                self.cache = pickle.load(f)
        else:
            # Create empty cache
            self.cache = {}

    def _get_cached_or_process(self, idx: int, modality: str):
        """
        Get cached data if available, otherwise process and cache.

        Args:
            idx: Sample index
            modality: Modality name

        Returns:
            processed_data: Processed data for the requested modality
        """
        if not self.cache_processed:
            # Process data without caching
            return self._process_modality(idx, modality)

        # Create cache key
        cache_key = f"{idx}_{modality}"

        if cache_key in self.cache:
            # Return cached data
            return self.cache[cache_key]
        else:
            # Process data and add to cache
            processed_data = self._process_modality(idx, modality)
            self.cache[cache_key] = processed_data
            return processed_data

    def _process_modality(self, idx: int, modality: str):
        """
        Process data for a specific modality.

        Args:
            idx: Sample index
            modality: Modality name

        Returns:
            processed_data: Processed data for the requested modality
        """
        sample = self.samples[idx]

        if modality == "visual":
            # Process visual data
            if "visual_path" not in sample or sample["visual_path"] is None:
                return None

            visual_path = sample["visual_path"]
            if self.preprocess:
                # Load and preprocess face image or video
                visual_data = preprocess_face_image(visual_path)
            else:
                # Just load the image or video
                if visual_path.suffix in [".jpg", ".png"]:
                    visual_data = cv2.imread(str(visual_path))
                    visual_data = cv2.cvtColor(visual_data, cv2.COLOR_BGR2RGB)
                else:
                    # Handle video
                    visual_data = []
                    cap = cv2.VideoCapture(str(visual_path))
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        visual_data.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cap.release()
                    visual_data = np.array(visual_data)

            # Apply transform if available
            if "visual" in self.transform:
                visual_data = self.transform["visual"](visual_data)

            return visual_data

        elif modality == "audio":
            # Process audio data
            if "audio_path" not in sample or sample["audio_path"] is None:
                return None

            audio_path = sample["audio_path"]
            if self.preprocess:
                # Load and preprocess audio
                audio_data, sr = preprocess_audio(audio_path)
            else:
                # Just load the audio
                audio_data, sr = librosa.load(str(audio_path), sr=None)

            # Apply transform if available
            if "audio" in self.transform:
                audio_data = self.transform["audio"](audio_data)

            return audio_data

        elif modality == "hr":
            # Process HR data
            if "hr_path" not in sample or sample["hr_path"] is None:
                return None

            hr_path = sample["hr_path"]
            if self.preprocess:
                # Load and preprocess HR signal
                hr_data = preprocess_hr_signal(hr_path)
            else:
                # Just load the HR signal
                hr_data = np.load(str(hr_path))

            # Apply transform if available
            if "hr" in self.transform:
                hr_data = self.transform["hr"](hr_data)

            return hr_data

        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def save_cache(self):
        """
        Save preprocessed data cache to disk.
        """
        if not self.cache_processed:
            return

        cache_file = self.cache_dir / f"{self.split}_cache.pkl"

        logging.info(f"Saving cache to {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(self.cache, f)

    def __len__(self) -> int:
        """
        Get number of samples in the dataset.

        Returns:
            Length of dataset
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            sample: Dictionary with modality data and label
        """
        sample = self.samples[idx]
        result = {}

        # Load modalities
        for modality in self.modalities:
            try:
                if modality == "visual" and "visual_path" in sample:
                    if sample["visual_path"] is not None:
                        result["visual"] = self._get_cached_or_process(idx, "visual")

                elif modality == "audio" and "audio_path" in sample:
                    if sample["audio_path"] is not None:
                        result["audio"] = self._get_cached_or_process(idx, "audio")

                elif modality == "hr" and "hr_path" in sample:
                    if sample["hr_path"] is not None:
                        result["hr"] = self._get_cached_or_process(idx, "hr")

                elif modality == "mocap" and "mocap_path" in sample:
                    if sample["mocap_path"] is not None:
                        result["mocap"] = self._get_cached_or_process(idx, "mocap")
            except Exception as e:
                logging.warning(f"Error loading {modality} for {sample.get('utterance_id', 'unknown')}: {e}")
                # Continue without this modality

        # Get emotion label
        emotion = sample["emotion"]
        label = self.emotion_map[emotion]
        result["labels"] = torch.tensor(label, dtype=torch.long)

        # Add metadata
        result["subject_id"] = sample.get("subject_id", -1)
        result["session_id"] = sample.get("session_id", -1)
        result["utterance_id"] = sample.get("utterance_id", "")

        # Ensure at least one modality is loaded
        if not any(m in result for m in self.modalities):
            logging.warning(f"No modalities loaded for sample {sample['utterance_id']}")
            # Return the simplest valid sample with dummy data for the first modality
            if self.modalities:
                first_mod = self.modalities[0]
                if first_mod == "visual":
                    result[first_mod] = torch.zeros((3, 224, 224), dtype=torch.float32)
                elif first_mod == "audio":
                    result[first_mod] = torch.zeros(16000, dtype=torch.float32)
                elif first_mod == "hr":
                    result[first_mod] = torch.zeros(2560, dtype=torch.float32)

        return result


class IEMOCAPDataset(EmotionDataset):
    """
    Dataset for the IEMOCAP (Interactive Emotional Dyadic Motion Capture) dataset.

    IEMOCAP contains multimodal recordings of dyadic interactions with emotional
    content, including audio, video, motion capture, and physiological signals.
    Each interaction is segmented into utterances with emotion annotations.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        modalities: List[str] = ["visual", "audio", "mocap"],
        transform: Optional[Dict[str, Callable]] = None,
        preprocess: bool = True,
        cache_processed: bool = True,
        segment_length: int = 3.0,
        emotion_map: Optional[Dict[str, int]] = None,
        val_sessions: List[int] = [5],  # Use session 5 for validation by default
        test_sessions: List[int] = [1],  # Use session 1 for testing by default
        include_improvised: bool = True,
        include_scripted: bool = True,
    ):
        """
        Initialize IEMOCAP dataset.

        Args:
            root_dir: Root directory of the IEMOCAP dataset
            split: Data split ("train", "val", "test")
            modalities: List of modalities to load
            transform: Dictionary of transform functions for each modality
            preprocess: Whether to preprocess data
            cache_processed: Whether to cache preprocessed data
            segment_length: Length of each segment in seconds
            emotion_map: Optional mapping from emotion names to indices.
                         If provided, it MUST use IEMOCAP codes ("neu", "hap", etc.) as keys.
                         If None, a default IEMOCAP mapping is used.
            val_sessions: Sessions to use for validation
            test_sessions: Sessions to use for testing
            include_improvised: Whether to include improvised scenarios
            include_scripted: Whether to include scripted scenarios
        """
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions
        self.include_improvised = include_improvised
        self.include_scripted = include_scripted

        # Define the standard IEMOCAP code -> index map
        default_iemocap_map = {
            "neu": 0,  # neutral
            "hap": 1,  # happiness
            "sad": 2,  # sadness
            "ang": 3,  # anger
            "fea": 4,  # fear
            "sur": 5,  # surprise
            "dis": 6,  # disgust
            "fru": 1,  # frustration (map to happiness - IEMOCAP-specific)
            "exc": 1,  # excitement (map to happiness - IEMOCAP-specific)
            "oth": 0,  # other (map to neutral)
        }

        # Use the provided map if available, otherwise use the default IEMOCAP map.
        # The provided map MUST use IEMOCAP codes ("neu", "hap", etc.) as keys.
        final_emotion_map = emotion_map if emotion_map is not None else default_iemocap_map

        super().__init__(
            root_dir=root_dir,
            split=split,
            modalities=modalities,
            transform=transform,
            preprocess=preprocess,
            cache_processed=cache_processed,
            segment_length=segment_length,
            emotion_map=final_emotion_map,  # Pass the correct map to the base class
        )
        # self.emotion_map in the base class now holds the correct map

    def _load_dataset(self):
        """
        Load IEMOCAP dataset samples.
        """
        self.samples = []

        # Find all session directories
        session_dirs = sorted(glob.glob(str(self.root_dir / "Session*")))

        if not session_dirs:
            logging.warning(f"No session directories found in {self.root_dir}")
            logging.warning(f"Directory contents: {os.listdir(self.root_dir)}")
            return

        logging.info(f"Found {len(session_dirs)} session directories: {session_dirs}")

        for session_dir in session_dirs:
            # Extract session number
            session_id = int(os.path.basename(session_dir)[-1])

            # Determine if this session belongs to the current split
            if (
                self.split == "val"
                and session_id in self.val_sessions
                or self.split == "test"
                and session_id in self.test_sessions
                or self.split == "train"
                and session_id not in self.val_sessions + self.test_sessions
            ):
                # Load session data
                self._load_session(session_dir, session_id)

    def _load_session(self, session_dir: str, session_id: int):
        """
        Load data for a specific IEMOCAP session.

        Args:
            session_dir: Session directory path
            session_id: Session ID
        """
        session_dir = Path(session_dir)
        logging.debug(f"Loading session {session_id} from {session_dir}")

        # Key paths in IEMOCAP structure - based on exploration output
        emo_eval_dir = session_dir / "dialog" / "EmoEvaluation" / "Categorical"
        sentences_dir = session_dir / "sentences" / "wav"
        video_dir = session_dir / "dialog" / "avi" / "DivX"
        mocap_dir = session_dir / "sentences" / "MOCAP_rotated"

        # Check if critical directories exist
        if not emo_eval_dir.exists():
            logging.warning(f"Emotion evaluation directory not found: {emo_eval_dir}")
            # Try alternate location
            emo_eval_dir = session_dir / "dialog" / "EmoEvaluation"
            if not emo_eval_dir.exists():
                logging.error(f"No emotion evaluation directory found in session {session_id}")
                return

        # Load emotion labels (categorical)
        label_files = sorted(glob.glob(str(emo_eval_dir / "*.txt")))

        # If no files found directly, look in subdirectories
        if not label_files:
            for subdir in emo_eval_dir.glob("**/"):
                if subdir.is_dir():
                    subdir_files = list(subdir.glob("*.txt"))
                    if subdir_files:
                        label_files.extend([str(f) for f in subdir_files])
                        logging.info(f"Found {len(subdir_files)} label files in {subdir}")

        if not label_files:
            logging.warning(f"No label files found for session {session_id}")
            return

        logging.info(f"Found {len(label_files)} emotion label files in session {session_id}")

        samples_added = 0
        total_utterances_checked = 0
        skipped_no_emotion = 0
        skipped_unwanted_emotion = 0

        for label_file in label_files:
            # Get dialog ID from filename, removing the evaluator suffix (_e*_cat)
            full_filename = Path(label_file).stem

            # Extract the base dialog ID by removing the evaluator suffix
            # Pattern examples: Ses01M_script03_2_e4_cat, Ses01F_impro07_e3_cat
            dialog_id = full_filename

            # Check if dialog_id has evaluator suffix and remove it
            if "_e" in dialog_id and dialog_id.split("_e")[-1].split("_")[0].isdigit():
                # Split at the _e* pattern and take the first part
                dialog_id = dialog_id.split("_e")[0]

            logging.debug(f"Processing dialog: {dialog_id} (from label file {full_filename})")

            # Determine if it's improvised or scripted
            is_improvised = "impro" in dialog_id
            is_scripted = "script" in dialog_id

            # Skip based on inclusion flags
            if (is_improvised and not self.include_improvised) or (is_scripted and not self.include_scripted):
                logging.debug(f"Skipping dialog {dialog_id} (is_improvised={is_improvised}, is_scripted={is_scripted})")
                continue

            # Parse emotion labels
            emotion_dict, dimensional_dict = self._parse_emotion_labels(label_file)
            if not emotion_dict:
                logging.warning(f"No emotions found in {label_file}")
                continue

            logging.debug(f"Found {len(emotion_dict)} emotion labels for dialog {dialog_id}")

            # IMPORTANT: In IEMOCAP, audio files are in nested directories by dialog ID
            dialog_wav_dir = sentences_dir / dialog_id
            if not dialog_wav_dir.exists():
                logging.warning(f"Dialog wav directory not found: {dialog_wav_dir}")
                continue

            # Find all WAV files for this dialog
            utterance_files = sorted(glob.glob(str(dialog_wav_dir / "*.wav")))

            if not utterance_files:
                logging.warning(f"No utterance files found for dialog {dialog_id}")
                continue

            logging.debug(f"Found {len(utterance_files)} utterances for dialog {dialog_id}")
            total_utterances_checked += len(utterance_files)

            for utterance_path in utterance_files:
                # Skip tiny/hidden files that might be MacOS artifacts
                if Path(utterance_path).stat().st_size < 1000:
                    logging.debug(f"Skipping small file: {utterance_path}")
                    continue

                utterance_id = Path(utterance_path).stem

                # Check if emotion label exists
                if utterance_id not in emotion_dict:
                    skipped_no_emotion += 1
                    continue

                emotion = emotion_dict[utterance_id]

                # Skip utterances with unwanted emotions
                if emotion not in self.emotion_map:
                    skipped_unwanted_emotion += 1
                    logging.debug(f"Skipping unwanted emotion '{emotion}' for {utterance_id}")
                    continue

                # Video is at the dialog level in IEMOCAP (one video per dialog)
                video_path = video_dir / f"{dialog_id}.avi"
                video_exists = video_path.exists()

                # Check if this utterance has MoCap data - they're in folders by dialog ID
                mocap_path = None
                dialog_mocap_dir = mocap_dir / dialog_id
                if dialog_mocap_dir.exists():
                    possible_mocap_path = dialog_mocap_dir / f"{utterance_id}.txt"
                    if possible_mocap_path.exists():
                        mocap_path = possible_mocap_path

                # Create sample
                sample = {
                    "subject_id": session_id,
                    "session_id": session_id,
                    "dialog_id": dialog_id,
                    "utterance_id": utterance_id,
                    "emotion": emotion,
                    "valence": dimensional_dict.get(utterance_id, {}).get("valence", 0.0),
                    "activation": dimensional_dict.get(utterance_id, {}).get("activation", 0.0),
                    "dominance": dimensional_dict.get(utterance_id, {}).get("dominance", 0.0),
                    "audio_path": str(utterance_path),
                    "visual_path": str(video_path) if video_exists else None,
                    "mocap_path": str(mocap_path) if mocap_path else None,
                    "hr_path": None,  # IEMOCAP doesn't have HR data as shown in exploration
                    "has_mocap": mocap_path is not None,
                    "actor_gender": utterance_id.split("_")[-1][0] if "_" in utterance_id else "?",
                    "is_improvised": is_improvised,
                }

                self.samples.append(sample)
                samples_added += 1

        logging.info(
            f"Session {session_id} summary: Added {samples_added} samples, "
            f"checked {total_utterances_checked} utterances, "
            f"skipped {skipped_no_emotion} without emotion label, "
            f"skipped {skipped_unwanted_emotion} with unwanted emotions"
        )

    def _parse_emotion_labels(self, label_file: str) -> tuple:
        """
        Parse categorical emotion labels from IEMOCAP label file.
        Format seen in exploration: <utterance_id> :<Emotion>; ()

        Args:
            label_file: Path to label file

        Returns:
            emotion_dict: Dictionary mapping utterance IDs to emotion codes
            dimensional_dict: Empty dictionary (dimensional parsing removed for simplicity)
        """
        emotion_dict = {}
        dimensional_dict = {}
        lines_processed = 0
        emotions_found = 0

        try:
            with open(label_file, "r") as f:
                for line_num, line in enumerate(f):
                    lines_processed += 1
                    line = line.strip()

                    # Skip empty lines or comment lines
                    if not line or line.startswith("#") or line.startswith("*"):
                        continue

                    # Split the line at the first colon
                    parts = line.split(":", 1)  # Split only once at the first colon
                    if len(parts) == 2:
                        utterance_id = parts[0].strip()
                        emotion_part = parts[1].strip()

                        # Basic validation of utterance ID format
                        if not utterance_id.startswith("Ses"):
                            logging.warning(
                                f"[{os.path.basename(label_file)}:{line_num + 1}] Skipping line, unexpected utterance ID format: {line}"
                            )
                            continue

                        # Extract the emotion label (before any semicolon)
                        emotion = emotion_part.split(";")[0].strip()

                        # Map full emotion names to IEMOCAP codes based on exploration output
                        iemocap_label_to_code = {
                            "Neutral state": "neu",
                            "Neutral": "neu",
                            "Happiness": "hap",
                            "Sadness": "sad",
                            "Anger": "ang",
                            "Fear": "fea",
                            "Surprise": "sur",
                            "Disgust": "dis",
                            "Frustration": "fru",
                            "Excited": "exc",
                            "Other": "oth",
                            # Keep short codes too in case they appear
                            "neu": "neu",
                            "hap": "hap",
                            "sad": "sad",
                            "ang": "ang",
                            "fea": "fea",
                            "sur": "sur",
                            "dis": "dis",
                            "fru": "fru",
                            "exc": "exc",
                            "oth": "oth",
                        }

                        emotion_code = iemocap_label_to_code.get(emotion, None)

                        if emotion_code and emotion_code in self.emotion_map:
                            if utterance_id in emotion_dict:
                                logging.debug(
                                    f"[{os.path.basename(label_file)}:{line_num + 1}] Utterance ID {utterance_id} already has emotion {emotion_dict[utterance_id]}. Overwriting with {emotion_code}."
                                )
                            emotion_dict[utterance_id] = emotion_code
                            emotions_found += 1
                            logging.debug(
                                f"[{os.path.basename(label_file)}:{line_num + 1}] Parsed emotion '{emotion_code}' (from '{emotion}') for {utterance_id}"
                            )
                        else:
                            logging.debug(
                                f"[{os.path.basename(label_file)}:{line_num + 1}] Skipping unknown/invalid emotion '{emotion}' for {utterance_id}: {line}"
                            )
                    else:
                        logging.warning(
                            f"[{os.path.basename(label_file)}:{line_num + 1}] Skipping line, unexpected format (no colon?): {line}"
                        )

        except FileNotFoundError:
            logging.error(f"Label file not found: {label_file}")
            return {}, {}
        except Exception as e:
            logging.error(f"Error reading or parsing label file {label_file}: {e}")
            return {}, {}

        logging.debug(
            f"Finished parsing {label_file}. Processed {lines_processed} lines. Parsed {emotions_found} emotions for {len(emotion_dict)} unique utterance IDs."
        )
        return emotion_dict, dimensional_dict


class RAVDESSDataset(EmotionDataset):
    """
    Dataset for the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

    The RAVDESS dataset contains 7356 files from 24 actors (12 female, 12 male) expressing
    different emotions in both speech and song, with multiple modalities (audio-only,
    video-only, and audio-video).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        modalities: List[str] = ["visual", "audio"],
        transform: Optional[Dict[str, Callable]] = None,
        preprocess: bool = True,
        cache_processed: bool = True,
        segment_length: int = 3.0,
        emotion_map: Optional[Dict[str, int]] = None,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        include_song: bool = True,
        include_speech: bool = True,
        seed: int = 42,
    ):
        """
        Initialize RAVDESS dataset.

        Args:
            root_dir: Root directory of the RAVDESS dataset
            split: Data split ("train", "val", "test")
            modalities: List of modalities to load
            transform: Dictionary of transform functions for each modality
            preprocess: Whether to preprocess data
            cache_processed: Whether to cache preprocessed data
            segment_length: Length of each segment in seconds
            emotion_map: Mapping from emotion names to indices
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
            include_song: Whether to include song files
            include_speech: Whether to include speech files
            seed: Random seed for reproducibility
        """
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.include_song = include_song
        self.include_speech = include_speech

        # RAVDESS-specific emotion mapping
        self.ravdess_emotion_map = emotion_map or {
            "neutral": 0,
            "calm": 7,  # map to contempt
            "happiness": 1,
            "sadness": 2,
            "anger": 3,
            "fear": 4,
            "disgust": 5,
            "surprise": 6,
        }

        super().__init__(
            root_dir=root_dir,
            split=split,
            modalities=modalities,
            transform=transform,
            preprocess=preprocess,
            cache_processed=cache_processed,
            segment_length=segment_length,
            emotion_map=self.ravdess_emotion_map,  # Use the RAVDESS-specific map
        )

    def _load_dataset(self):
        """
        Load RAVDESS dataset samples.
        """
        # Find all actor directories
        actor_dirs = sorted(glob.glob(str(self.root_dir / "Actor_*")))
        if not actor_dirs:
            # Try alternative structure with audio and video folders
            all_files = []

            # Check for Audio-only files structure
            audio_speech_path = self.root_dir / "Audio_Speech_Actors_01-24"
            audio_song_path = self.root_dir / "Audio_Song_Actors_01-24"

            # Check for Video folder structure
            video_dirs = sorted(glob.glob(str(self.root_dir / "Video_*_Actor_*")))

            # If neither structure exists, try to find files directly in root
            if not audio_speech_path.exists() and not audio_song_path.exists() and not video_dirs:
                audio_files = glob.glob(str(self.root_dir / "**/*.wav"), recursive=True)
                video_files = glob.glob(str(self.root_dir / "**/*.mp4"), recursive=True)
                all_files = audio_files + video_files
            else:
                # Handle Audio-only files
                if self.include_speech and audio_speech_path.exists():
                    all_files.extend(glob.glob(str(audio_speech_path / "*.wav")))

                if self.include_song and audio_song_path.exists():
                    all_files.extend(glob.glob(str(audio_song_path / "*.wav")))

                # Handle Video files (both Audio-Video and Video-only)
                for video_dir in video_dirs:
                    if ("Speech" in video_dir and self.include_speech) or ("Song" in video_dir and self.include_song):
                        all_files.extend(glob.glob(str(Path(video_dir) / "*.mp4")))
        else:
            # Actor-based folder structure
            all_files = []
            for actor_dir in actor_dirs:
                # Get all audio and video files in the actor directory
                if self.include_speech:
                    all_files.extend(glob.glob(str(Path(actor_dir) / "*-01-*.wav")))  # Speech audio
                    all_files.extend(glob.glob(str(Path(actor_dir) / "*-01-*.mp4")))  # Speech video

                if self.include_song:
                    all_files.extend(glob.glob(str(Path(actor_dir) / "*-02-*.wav")))  # Song audio
                    all_files.extend(glob.glob(str(Path(actor_dir) / "*-02-*.mp4")))  # Song video

        # Create samples
        all_samples = []

        for file_path in all_files:
            # Get filename without extension
            file_name = os.path.basename(file_path)
            file_id = file_name.split(".")[0]

            # Parse filename components
            # Format: xx-xx-xx-xx-xx-xx-xx.wav/mp4
            # Modality-Vocal channel-Emotion-Intensity-Statement-Repetition-Actor
            parts = file_id.split("-")
            if len(parts) != 7:
                continue

            modality_code = parts[0]  # 01=AV, 02=video-only, 03=audio-only
            vocal_code = parts[1]  # 01=speech, 02=song
            emotion_code = parts[2]  # emotion code
            intensity_code = parts[3]  # 01=normal, 02=strong
            # repetition_code = parts[5]  # 01/02 = repetition number
            actor_id = int(parts[6])  # actor ID

            # Map emotion code to emotion name
            emotion_map = {
                "01": "neutral",
                "02": "calm",
                "03": "happiness",
                "04": "sadness",
                "05": "anger",
                "06": "fear",
                "07": "disgust",
                "08": "surprise",
            }

            emotion = emotion_map.get(emotion_code, "unknown")

            # Skip files with unwanted emotions
            if emotion not in self.ravdess_emotion_map:
                continue

            # Determine if this is audio or video
            is_audio = file_path.endswith(".wav") or modality_code == "03"
            is_video = file_path.endswith(".mp4") and modality_code in ["01", "02"]

            # Create sample
            sample = {
                "subject_id": actor_id,
                "session_id": -1,  # RAVDESS doesn't have sessions
                "utterance_id": file_id,
                "emotion": emotion,
                "audio_path": file_path if is_audio else None,
                "visual_path": file_path if is_video else None,
                "hr_path": None,  # RAVDESS doesn't have HR data
                "vocal_type": "speech" if vocal_code == "01" else "song",
                "intensity": "normal" if intensity_code == "01" else "strong",
            }

            all_samples.append(sample)

        # Split data into train/val/test
        random.seed(self.seed)
        all_samples = sorted(all_samples, key=lambda x: x["utterance_id"])
        random.shuffle(all_samples)

        num_samples = len(all_samples)
        num_test = int(num_samples * self.test_ratio)
        num_val = int(num_samples * self.val_ratio)
        num_train = num_samples - num_test - num_val

        # Select samples for the current split
        if self.split == "train":
            self.samples = all_samples[:num_train]
        elif self.split == "val":
            self.samples = all_samples[num_train : num_train + num_val]
        elif self.split == "test":
            self.samples = all_samples[num_train + num_val :]
        else:
            self.samples = all_samples


class CombinedEmotionDataset(EmotionDataset):
    """
    Combined dataset that merges multiple emotion datasets.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize combined dataset.

        Args:
            datasets: List of datasets to combine
            weights: Optional weights for sampling from each dataset
        """
        self.datasets = datasets
        self.weights = weights

        # Normalize weights if provided
        if weights is not None:
            weights_sum = sum(weights)
            self.weights = [w / weights_sum for w in weights]

        # Determine total length
        dataset_lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = np.cumsum(dataset_lengths)
        self.total_length = sum(dataset_lengths)

        # Map indices to dataset and index within dataset
        self.index_map = []
        for dataset_idx, length in enumerate(dataset_lengths):
            for sample_idx in range(length):
                self.index_map.append((dataset_idx, sample_idx))

    def __len__(self) -> int:
        """
        Get number of samples in the combined dataset.

        Returns:
            Length of combined dataset
        """
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the combined dataset.

        Args:
            idx: Sample index

        Returns:
            sample: Dictionary with modality data and label
        """
        dataset_idx, sample_idx = self.index_map[idx]
        return self.datasets[dataset_idx][sample_idx]


def create_emotion_dataloaders(
    iemocap_dir: Optional[str] = None,
    ravdess_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    modalities: List[str] = ["visual", "audio", "hr"],
    transform: Optional[Dict[str, Callable]] = None,
    preprocess: bool = True,
    cache_processed: bool = True,
    segment_length: int = 3.0,
    emotion_map: Optional[Dict[str, int]] = None,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for emotion recognition datasets.

    Args:
        iemocap_dir: Path to IEMOCAP dataset (optional)
        ravdess_dir: Path to RAVDESS dataset (optional)
        batch_size: Batch size
        num_workers: Number of workers for data loading
        modalities: List of modalities to load
        transform: Dictionary of transform functions for each modality
        preprocess: Whether to preprocess data
        cache_processed: Whether to cache preprocessed data
        segment_length: Length of each segment in seconds
        emotion_map: Mapping from emotion names to indices

    Returns:
        dataloaders: Dictionary of dataloaders for train, val, test splits
    """
    # Validate dataset directories
    if iemocap_dir is not None and not os.path.isdir(iemocap_dir):
        raise ValueError(f"IEMOCAP directory '{iemocap_dir}' does not exist or is not a directory.")
    if ravdess_dir is not None and not os.path.isdir(ravdess_dir):
        raise ValueError(f"RAVDESS directory '{ravdess_dir}' does not exist or is not a directory.")

    train_datasets = []
    val_datasets = []
    test_datasets = []

    # IEMOCAP dataset
    if iemocap_dir is not None:
        iemocap_train = IEMOCAPDataset(
            root_dir=iemocap_dir,
            split="train",
            modalities=modalities,
            transform=transform,
            preprocess=preprocess,
            cache_processed=cache_processed,
            segment_length=segment_length,
            emotion_map=emotion_map,
        )

        iemocap_val = IEMOCAPDataset(
            root_dir=iemocap_dir,
            split="val",
            modalities=modalities,
            transform=transform,
            preprocess=preprocess,
            cache_processed=cache_processed,
            segment_length=segment_length,
            emotion_map=emotion_map,
        )

        iemocap_test = IEMOCAPDataset(
            root_dir=iemocap_dir,
            split="test",
            modalities=modalities,
            transform=transform,
            preprocess=preprocess,
            cache_processed=cache_processed,
            segment_length=segment_length,
            emotion_map=emotion_map,
        )

        train_datasets.append(iemocap_train)
        val_datasets.append(iemocap_val)
        test_datasets.append(iemocap_test)

    # RAVDESS dataset (skip if hr modality is requested, since RAVDESS has no HR data)
    if ravdess_dir is not None and "hr" not in modalities:
        ravdess_train = RAVDESSDataset(
            root_dir=ravdess_dir,
            split="train",
            modalities=modalities,
            transform=transform,
            preprocess=preprocess,
            cache_processed=cache_processed,
            segment_length=segment_length,
            emotion_map=emotion_map,
        )

        ravdess_val = RAVDESSDataset(
            root_dir=ravdess_dir,
            split="val",
            modalities=modalities,
            transform=transform,
            preprocess=preprocess,
            cache_processed=cache_processed,
            segment_length=segment_length,
            emotion_map=emotion_map,
        )

        ravdess_test = RAVDESSDataset(
            root_dir=ravdess_dir,
            split="test",
            modalities=modalities,
            transform=transform,
            preprocess=preprocess,
            cache_processed=cache_processed,
            segment_length=segment_length,
            emotion_map=emotion_map,
        )

        train_datasets.append(ravdess_train)
        val_datasets.append(ravdess_val)
        test_datasets.append(ravdess_test)

    # Combine datasets if multiple are provided
    if len(train_datasets) > 1:
        train_dataset = CombinedEmotionDataset(train_datasets)
        val_dataset = CombinedEmotionDataset(val_datasets)
        test_dataset = CombinedEmotionDataset(test_datasets)
    elif len(train_datasets) == 1:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]
        test_dataset = test_datasets[0]
    else:
        raise ValueError("No datasets provided")

    # Log dataset source information
    if iemocap_dir is not None:
        iemocap_samples = sum(len(ds) for ds in train_datasets if isinstance(ds, IEMOCAPDataset))
        logging.info(f"IEMOCAP samples in training set: {iemocap_samples}")

    if ravdess_dir is not None:
        ravdess_samples = sum(len(ds) for ds in train_datasets if isinstance(ds, RAVDESSDataset))
        logging.info(f"RAVDESS samples in training set: {ravdess_samples}")

    # Ensure each split has samples
    for split_name, ds in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
        if len(ds) == 0:
            # List directory contents of dataset roots for debugging
            # Handle combined dataset or single dataset
            sub_datasets = getattr(ds, "datasets", [ds])
            root_dirs = []
            for sub in sub_datasets:
                root = getattr(sub, "root_dir", None)
                root_dirs.append(str(root) if root is not None else repr(sub))
            listings = {}
            for root in root_dirs:
                try:
                    listings[root] = os.listdir(root)
                except Exception as e:
                    listings[root] = f"Error listing dir: {e}"

            # Get the list of requested modalities
            modality_str = ", ".join(modalities)

            # Look for common issues
            potential_issues = []
            if "hr" in modalities:
                potential_issues.append("'hr' modality requested but IEMOCAP may not have HR data")
            if "mocap" in modalities and "mocap" not in getattr(sub_datasets[0], "modalities", []):
                potential_issues.append("'mocap' modality requested but not included in dataset configuration")

            issues_str = " | ".join(potential_issues) if potential_issues else "No specific issues identified"

            raise ValueError(
                f"No {split_name} samples found in dataset roots {root_dirs}. "
                f"Directory contents: {listings}. "
                f"Requested modalities: {modality_str}. "
                f"Potential issues: {issues_str}. "
                "Please check your dataset paths and modalities."
            )
        else:
            logging.info(f"Found {len(ds)} samples for {split_name} split")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def collate_fn(batch):
    """
    Custom collate function that handles None values and pads sequences.

    Args:
        batch: List of samples from the dataset

    Returns:
        Collated batch with padded sequences and proper handling of None values
    """
    # Print detailed batch information for debugging
    print(f"\n===== Collating batch with {len(batch)} samples =====")

    # Filter out None values for each key
    result = {}

    # Get all keys from all samples
    keys = set()
    for sample in batch:
        keys.update(sample.keys())

    # Process each key
    for key in sorted(list(keys)):  # Sort keys for consistent logging
        # Collect non-None values for this key
        values = [sample[key] for sample in batch if key in sample and sample[key] is not None]

        # Log key and number of non-None values
        print(f"Key: {key}, Non-None values: {len(values)}")

        # If no valid values, set to None
        if len(values) == 0:
            result[key] = None
            print(f"  No valid data for key {key}")
            continue

        # --- START NEW PADDING/STACKING LOGIC ---
        # Check if padding and stacking is needed (primarily for modality data)
        needs_padding_stacking = False
        is_modality_data = key in ["visual", "audio", "hr"]
        first_item = values[0]

        if is_modality_data and isinstance(first_item, (torch.Tensor, np.ndarray)):
            # Check if shapes differ
            first_shape = first_item.shape
            if not all(v.shape == first_shape for v in values if hasattr(v, "shape")):
                needs_padding_stacking = True

        if needs_padding_stacking:
            print(f"  Padding/stacking required for key: {key}")
            padded_tensors = []
            max_len = 0
            target_dim = -1  # Dimension to pad (usually the last one for time/sequence)

            # Determine max length and convert to tensors
            valid_items = []
            for i, v in enumerate(values):
                item_tensor = None
                if isinstance(v, np.ndarray):
                    try:
                        item_tensor = torch.from_numpy(v).float()
                    except Exception as e:
                        print(f"    Warning: Could not convert numpy array {i} for key {key}: {e}")
                        continue
                elif isinstance(v, torch.Tensor):
                    item_tensor = v.float()  # Ensure float
                else:
                    print(f"    Warning: Unexpected type {type(v)} at index {i} for key {key}")
                    continue  # Skip unexpected types

                if item_tensor is not None:
                    if item_tensor.ndim == 0:
                        print(f"    Warning: 0-dim tensor found at index {i} for key {key}. Skipping.")
                        continue
                    valid_items.append(item_tensor)
                    current_len = item_tensor.size(target_dim)
                    if current_len > max_len:
                        max_len = current_len

            # Pad and collect tensors
            if not valid_items:
                print(f"  No valid items to pad/stack for key {key}")
                result[key] = None
                continue

            print(f"    Max length for key {key}: {max_len}")
            for i, item_tensor in enumerate(valid_items):
                current_len = item_tensor.size(target_dim)
                pad_width = max_len - current_len
                if pad_width > 0:
                    # Create padding tuple (pad_left, pad_right, pad_top, pad_bottom, ...)
                    # We only want to pad the target_dim (last dimension)
                    padding_dims = [0] * (item_tensor.ndim * 2)
                    padding_dims[0] = pad_width  # Pad right side of the last dimension
                    try:
                        # Use tuple(reversed(padding_dims)) for F.pad
                        padded_item = F.pad(item_tensor, tuple(reversed(padding_dims)), mode="constant", value=0)
                        padded_tensors.append(padded_item)
                    except Exception as e:
                        print(f"    Error padding tensor {i} with shape {item_tensor.shape} for key {key}: {e}")
                        # Fallback: skip this item
                elif pad_width == 0:
                    padded_tensors.append(item_tensor)
                else:  # Should not happen
                    print(f"    Warning: Negative padding width for item {i}, key {key}? Skipping.")

            # Stack the padded tensors
            if padded_tensors:
                try:
                    result[key] = torch.stack(padded_tensors)
                    print(f"  Successfully padded/stacked {len(padded_tensors)} items for key {key}: {result[key].shape}")
                except Exception as e:
                    print(f"  Error stacking padded tensors for key {key}: {e}")
                    # Fallback: return the list of padded tensors (less ideal)
                    result[key] = padded_tensors
            else:
                print(f"  No tensors left after padding attempt for key {key}")
                result[key] = None
        # --- END NEW PADDING/STACKING LOGIC ---
        else:
            # --- UNIFORM SHAPE OR NON-MODALITY DATA ---
            try:
                if isinstance(first_item, torch.Tensor):
                    result[key] = torch.stack(values)
                    if is_modality_data or key == "labels":
                        print(f"  Stacked uniform tensors for key {key}: {result[key].shape}")
                elif isinstance(first_item, np.ndarray):
                    tensor_vals = [torch.from_numpy(v).float() for v in values]
                    result[key] = torch.stack(tensor_vals)
                    if is_modality_data:
                        print(f"  Stacked uniform numpy arrays for key {key}: {result[key].shape}")
                elif isinstance(first_item, (int, float)):
                    result[key] = torch.tensor(values)
                elif isinstance(first_item, (str, bool)):
                    # Cannot convert str/bool directly usually, keep as list
                    result[key] = values
                else:
                    # Keep as list for other types
                    print(f"  Keeping key {key} as list (type: {type(first_item)})")
                    result[key] = values
            except Exception as e:
                print(f"  Error processing key {key} (type: {type(first_item)}): {e}. Keeping as list.")
                # Fallback to list if any conversion/stacking fails
                result[key] = values

    # Final checks on batch structure
    if "labels" in result and result["labels"] is not None:
        if isinstance(result["labels"], torch.Tensor):
            print(f"Final batch labels shape: {result['labels'].shape}")
        else:
            print(f"Final batch labels is not a tensor: {type(result['labels'])}")
    else:
        print("WARNING: Batch has no labels!")

    for key in ["visual", "audio", "hr"]:
        if key in result and result[key] is not None:
            if isinstance(result[key], torch.Tensor):
                print(f"Final batch {key} shape: {result[key].shape}")
            else:
                print(f"Final batch {key} is not a tensor: {type(result[key])}")
        else:
            print(f"Final batch has no {key} data")

    print("=====================================")

    return result
