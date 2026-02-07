import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import utils.logutils as log

class MoodLatentMLP(nn.Module):
    """
    Simple Feed-Forward Neural Network (MLP).
    
    It maps low-level audio features (Input) to the GAN's Latent Space (Output).
    Structure: Input -> [Linear->ReLU->BatchNorm] x 2 -> Linear->ReLU -> Output
    """
    def __init__(self, input_size=5, output_size=512):
        super(MoodLatentMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.network(x)

class MoodPredictor:
    """
    Wrapper class to handle Model loading, Input Scaling, and Inference.
    """
    def __init__(self, model_path, scaler_path, use_gpu=True):
        """
        Initializes the predictor by loading the model and the scaler.
        """

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            log.info("MLP Manager: GPU mode (CUDA) activated.")
        else:
            self.device = torch.device('cpu')
            if use_gpu and not torch.cuda.is_available():
                log.warning("MLP Manager: GPU requested but not found. Fallback to CPU.")
            else:
                log.info("MLP Manager: CPU mode forced.")

        self.model_loaded = False
        
        # Check if files exist
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            log.error(f"ERROR: Model or scaler files not found:\nModel: {model_path}\nScaler: {scaler_path}")
            return

        try:
            # Load Scaler
            self.scaler = joblib.load(scaler_path)
            
            # Load Model
            self.model = MoodLatentMLP(input_size=5, output_size=512)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            log.success(f"MLP model 'MoodPredictor' initialized on {self.device}.")
            
        except Exception as e:
            log.error(f"ERROR loading the model: {e}")
            self.model_loaded = False

    def get_latent_vector(self, contrast: float, flatness: float, onset: float, zrc: float, chroma_var: float):
        """
        Takes raw audio features, normalizes them, and returns the latent vector.
        
        Args:
            contrast, flatness, onset, zrc, chroma_var (float): Extracted audio features.

        Returns:
            np.array: A 512-dimensional vector (float32).
        """
        if not self.model_loaded:
            # Return empty vector or zeros in case of error
            return np.zeros(512, dtype=np.float32)

        # Input Preparation (shape [1, 5])
        # Input must be a 2D array for the scaler and PyTorch
        raw_input = np.array([[contrast, flatness, onset, zrc, chroma_var]], dtype=np.float32)

        # Normalization (Using the scaler learned during training)
        try:
            scaled_input = self.scaler.transform(raw_input)
        except Exception as e:
            log.error(f"Scaling error: {e}")
            return np.zeros(512, dtype=np.float32)

        # Conversion to Tensor
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).to(self.device)

        # Inference
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # Output Conversion to Numpy
        latent_vector = output_tensor.cpu().numpy().flatten()

        return latent_vector