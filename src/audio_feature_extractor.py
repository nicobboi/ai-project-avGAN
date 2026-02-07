import numpy as np
import librosa
from dataclasses import dataclass
import utils.logutils as log

@dataclass
class AudioAnalysis:
    """
    Data structure to hold the extracted audio features for a specific frame.
    """
    spectral_contrast: float
    spectral_flatness: float
    onset_strength: float
    zero_crossing_rate: float
    chroma_variance: float

class AudioFeatureExtractor:
    """
    Handles the extraction of spectral features from raw audio data using Librosa.
    
    This class also manages the pre-computation of feature ranges (min/max percentiles)
    for a specific audio file to normalize values effectively during playback.
    """

    def __init__(self):
        self.features_ranges = None


    def process_audio(self, audio_data: np.ndarray, sr: int) -> AudioAnalysis:
        """
        Extracts features from a single audio chunk in real-time.
        
        Args:
            audio_data (np.ndarray): The raw floating-point audio samples.
            sr (int): Sample rate.
            
        Returns:
            AudioAnalysis: Dataclass containing the computed features, or empty values if silent.
        """
        if len(audio_data) == 0: return self._get_empty_features()
        y = audio_data.astype(np.float32)
        
        # Silence detection (threshold 0.005)
        if np.sqrt(np.mean(y**2)) < 0.005: return self._get_empty_features()

        n_samples = len(y)
        n_fft = min(n_samples, 1024) 
        if n_fft < 128: return self._get_empty_features()
        hop_length = n_fft // 4

        try:
            # Optimization: Short-time Fourier Transform executed once for the whole chunk
            S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) 
            
            # Analysis of the difference between energy peaks and valleys in the spectrum, 
            # indicates sound definition/texture
            contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=6) 
            flatness = librosa.feature.spectral_flatness(S=S) 
            onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
            chroma = librosa.feature.chroma_stft(S=S, sr=sr, tuning=0.0)

            return AudioAnalysis(
                self._round_feature_value('spectral_contrast', np.mean(contrast[1:, :]), 2),
                self._round_feature_value('spectral_flatness', np.mean(flatness), 4),
                self._round_feature_value('onset_strength', np.mean(onset_env), 3),
                self._round_feature_value('zero_crossing_rate', np.mean(zcr), 4),
                self._round_feature_value('chroma_variance', np.mean(np.var(chroma, axis=1)), 4)
            )
        except: return AudioFeatureExtractor._get_empty_features()

    def compute_features_ranges(self, y: np.ndarray, sr: int) -> None:
        """
        Pre-computes the dynamic range (10th and 90th percentiles) for each feature
        across the entire audio file.
        
        This allows `process_audio` to return normalized/clipped values relative 
        to the specific song's dynamics, rather than absolute mathematical limits.
        """
        rows = []
        CHUNK_SIZE = 1024
        
        try:
            total_samples = len(y)
            # Calculate how many chunks there are
            num_chunks = int(np.ceil(total_samples / CHUNK_SIZE))
            
            for i in range(num_chunks):
                start = i * CHUNK_SIZE
                end = start + CHUNK_SIZE
                
                chunk = y[start:end]
                
                # Skip chunks that are too small (file tail)
                if len(chunk) < 512: 
                    continue

                # Feature extraction via the standard method
                features: AudioAnalysis = self.process_audio(chunk, sr)
                
                
                # If -1.0 is returned, it means silence or error. SKIP.
                if features.spectral_contrast == -1.0:
                    continue

                # Create data row
                row = {
                    'chunk_index': i,
                    'timestamp_sec': round(start / sr, 3),
                    'contrast': features.spectral_contrast,
                    'flatness': features.spectral_flatness,
                    'onset': features.onset_strength,
                    'zcr': features.zero_crossing_rate,
                    'chroma_var': features.chroma_variance
                }
                rows.append(row)
            
            # --- PERCENTILE CALCULATION ---
            if not rows:
                log.warning("No valid features extracted for range calculation.")
                return

            # Map between short keys used in 'row' and full keys expected by process_audio
            key_map = {
                'contrast': 'spectral_contrast',
                'flatness': 'spectral_flatness',
                'onset': 'onset_strength',
                'zcr': 'zero_crossing_rate',
                'chroma_var': 'chroma_variance'
            }

            computed_ranges = {}

            for short_key, full_key in key_map.items():
                # Extract all values for that specific feature from all rows
                values = [r[short_key] for r in rows]
                
                # Calculate percentiles
                pMin = np.percentile(values, 10)
                pMax = np.percentile(values, 90)
                
                # Save using the "full" key (e.g., spectral_contrast)
                computed_ranges[full_key] = (round(float(pMin), 4), round(float(pMax), 4))

            # Save state in the class
            self.features_ranges = computed_ranges
            log.info(f"Features ranges computed (from {len(rows)} chunks): {computed_ranges}")

        except Exception as e:
            log.error(f"Error extracting feature ranges: {e}")

    def _round_feature_value(self, feature_name: str, value: float, round_to: int = 2) -> float:
        """
        Internal helper: Clips the value to the pre-computed range (if available)
        and rounds it. This acts as a normalizer.
        """
        if self.features_ranges is None or feature_name not in self.features_ranges:
             return round(float(value), round_to)

        range_vals = self.features_ranges[feature_name]
        new_value = np.clip(value, range_vals[0], range_vals[1])
        
        return round(float(np.mean(new_value)), round_to)


    @staticmethod
    def _get_empty_features():
        """Returns a placeholder object for silence or errors."""
        return AudioAnalysis(-1.0, -1.0, -1.0, -1.0, -1.0)