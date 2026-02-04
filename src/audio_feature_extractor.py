import numpy as np
import librosa
from dataclasses import dataclass
import utils.logutils as log

@dataclass
class AudioAnalysis:
    spectral_contrast: float
    spectral_flatness: float
    onset_strength: float
    zero_crossing_rate: float
    chroma_variance: float

class AudioFeatureExtractor:

    def __init__(self):
        self.features_ranges = None


    def process_audio(self, audio_data: np.ndarray, sr: int) -> AudioAnalysis:
        if len(audio_data) == 0: return self._get_empty_features()
        y = audio_data.astype(np.float32)
        if np.sqrt(np.mean(y**2)) < 0.005: return self._get_empty_features()

        n_samples = len(y)
        n_fft = min(n_samples, 1024) 
        if n_fft < 128: return self._get_empty_features()
        hop_length = n_fft // 4

        try:
            S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) #ottimizzazione, trasformata di Fourier eseguita una sola volta per tutto il chunk
            contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=6) # Analisi differenza tra picchi di energia e valli dello spettro, permette di capire il livello di definizione del suono
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
        rows = []
        CHUNK_SIZE = 1024
        
        try:
            total_samples = len(y)
            # Calcolo quanti chunk ci sono
            num_chunks = int(np.ceil(total_samples / CHUNK_SIZE))
            
            for i in range(num_chunks):
                start = i * CHUNK_SIZE
                end = start + CHUNK_SIZE
                
                chunk = y[start:end]
                
                # Saltiamo chunk troppo piccoli (coda del file)
                if len(chunk) < 512: 
                    continue

                # Estrazione Feature tramite il metodo standard
                features: AudioAnalysis = self.process_audio(chunk, sr)
                
                
                # Se è tornato -1.0, significa silenzio o errore. SALTA.
                if features.spectral_contrast == -1.0:
                    continue

                # Creazione riga dati
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
            
            # --- CALCOLO PERCENTILI ---
            if not rows:
                log.warning("Nessuna feature valida estratta per il calcolo dei range.")
                return

            # Mappa tra le chiavi brevi usate nel row e le chiavi complete attese da process_audio
            key_map = {
                'contrast': 'spectral_contrast',
                'flatness': 'spectral_flatness',
                'onset': 'onset_strength',
                'zcr': 'zero_crossing_rate',
                'chroma_var': 'chroma_variance'
            }

            computed_ranges = {}

            for short_key, full_key in key_map.items():
                # Estraiamo tutti i valori per quella specifica feature da tutte le righe
                values = [r[short_key] for r in rows]
                
                # Calcolo percentili
                pMin = np.percentile(values, 10)
                pMax = np.percentile(values, 90)
                
                # Salviamo usando la chiave "full" (es. spectral_contrast)
                computed_ranges[full_key] = (round(float(pMin), 4), round(float(pMax), 4))

            # Salviamo lo stato nella classe
            self.features_ranges = computed_ranges
            log.info(f"Features ranges computed (from {len(rows)} chunks): {computed_ranges}")

        except Exception as e:
            log.error(f"Errore estrazione range feature: {e}")

    def _round_feature_value(self, feature_name: str, value: float, round_to: int = 2) -> float:
        if self.features_ranges is None or feature_name not in self.features_ranges:
             return round(float(value), round_to)

        range_vals = self.features_ranges[feature_name]
        new_value = np.clip(value, range_vals[0], range_vals[1])
        
        return round(float(np.mean(new_value)), round_to)


    @staticmethod
    def _get_empty_features():
        return AudioAnalysis(-1.0, -1.0, -1.0, -1.0, -1.0)