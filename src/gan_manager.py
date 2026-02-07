import torch
import os
import pickle
import numpy as np
import utils.logutils as log

# Optimizations for NVIDIA GPUs
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# DEVICE SELECTION (cuda if available, otherwise CPU)
class GANManager:
    """
    Manages the Generative Adversarial Network (GAN) model.

    Handles model loading, latent space interpolation, and image generation.
    It acts as the bridge between the numerical latent vectors (produced by the MLP)
    and the visual output.
    """
    def __init__(self, model_path, latent_dim=512, use_gpu=True):
        """
        Initializes the GAN manager.

        Args:
            model_path (str): Path to the pickled model network.
            latent_dim (int): Dimension of the latent space (z).
            use_gpu (bool): Flag to enable/disable GPU usage.
        """
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            log.info("GAN Manager: GPU mode (CUDA) activated.")
        else:
            self.device = torch.device('cpu')
            if use_gpu and not torch.cuda.is_available():
                log.warning("GAN Manager: GPU requested but not found. Fallback to CPU.")
            else:
                log.info("GAN Manager: CPU mode forced.")
        
        # Load model on video card and set 'eval' mode for memory saving
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)['G_ema'].to(self.device).eval() 
        
        # Identify the dimension of the latent vector
        self.latent_dim = self.model.z_dim if hasattr(self.model, 'z_dim') else latent_dim 
        
        # If the model is conditional (uses classes), create a zero vector 'c'.
        self.c = torch.zeros([1, self.model.c_dim]).to(self.device) if self.model.c_dim > 0 else None 
        
        self.reset_state() # Initialization of position in latent space

    def reset_state(self):
        """
        Creates a random starting point and an identical target to avoid 
        sudden jumps at startup.
        """
        self.current_z = torch.randn(1, self.latent_dim).to(self.device)
        self.target_z = self.current_z.clone()

    def get_distance_to_target(self):
        """
        Measures how far the current latent vector is from the target.
        Useful for debugging or UI visualization.
        """
        return torch.norm(self.target_z - self.current_z).item()

    def generate_image(self, audio_chunk, mlp_latent_vector) -> np.uint8:
        """
        Generates a single image frame based on audio dynamics and MLP suggestions.

        Args:
            audio_chunk (np.array): Raw audio data (used for RMS/energy calculation).
            mlp_latent_vector (np.array): Target latent vector predicted by the MLP.

        Returns:
            np.uint8: Generated image as a numpy array (H, W, C).
        """
        # Disable gradient calculation to save memory and speed.
        with torch.no_grad():
            # Audio energy calculation (RMS) to modulate transition speed.
            rms = np.linalg.norm(audio_chunk) / np.sqrt(len(audio_chunk)) if len(audio_chunk) > 0 else 0
            
            # STEP parameter: defines how much ground we cover towards the target in this frame.
            step = min(0.01 + (rms * 0.1), 0.2) 

            if mlp_latent_vector is not None:
                # New position suggested by the MLP
                new_target = torch.as_tensor(mlp_latent_vector, device=self.device).float().unsqueeze(0)
                # Spherical normalization
                self.target_z = new_target / (new_target.norm() + 1e-8) * np.sqrt(self.latent_dim)

            # 2. SPHERICAL INTERPOLATION (Slerp-like)
            # We use lerp to slide towards the target in a controlled manner
            self.current_z = torch.lerp(self.current_z, self.target_z, step)
            
            # 3. REDUCED DRIFT NOISE to remove excessive "jitter".
            self.current_z += torch.randn_like(self.current_z) * 0.003
            
            # 4. FP16 GENERATION (latent vector -> StyleGAN)
            # force_fp32=False enables Tensor Cores (doubled speed on RTX).
            img = self.model(self.current_z, self.c, force_fp32=False, noise_mode='const')
            
            # POST-PROCESSING: Reorder channels (H,W,C), scale colors (0-255), and convert to integers.
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8) 
            
            return img[0].cpu().numpy() # GPU -> RAM(CPU)