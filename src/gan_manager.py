import torch
import os
import pickle
import numpy as np
import utils.logutils as log

# Ottimizzazioni per GPU NVIDIA
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# SELEZIONE DEVICE (cuda se disponibile, se no CPU)
class GANManager:
    def __init__(self, model_path, latent_dim=512, use_gpu=True):
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            log.info("MLP Manager: Modalità GPU (CUDA) attivata.")
        else:
            self.device = torch.device('cpu')
            if use_gpu and not torch.cuda.is_available():
                log.warning("MLP Manager: GPU richiesta ma non trovata. Fallback su CPU.")
            else:
                log.info("MLP Manager: Modalità CPU forzata.")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)['G_ema'].to(self.device).eval() #caricamento modello su scheda video e impostazione modalità 'eval' per risparmio memoria
        self.latent_dim = self.model.z_dim if hasattr(self.model, 'z_dim') else latent_dim #identifica la dimensione del vettore latente
        self.c = torch.zeros([1, self.model.c_dim]).to(self.device) if self.model.c_dim > 0 else None # Se il modello è condizionale (usa classi), crea un vettore 'c' di zeri.
        self.reset_state() #inizializzazione della posizione nello spazio latente

    # Crea un punto di partenza casuale e un target identico, evita scatti all'avvio
    def reset_state(self):
        self.current_z = torch.randn(1, self.latent_dim).to(self.device)
        self.target_z = self.current_z.clone()

    def get_distance_to_target(self):
        # Misura quanto siamo lontani dall'obiettivo
        return torch.norm(self.target_z - self.current_z).item()

    def generate_image(self, audio_chunk, mlp_latent_vector) -> np.uint8:
        # Disabilita il calcolo dei gradienti per risparmiare memoria e velocità.
        with torch.no_grad():
            # Calcolo dell'energia audio (RMS) per modulare la velocità di transizione.
            rms = np.linalg.norm(audio_chunk) / np.sqrt(len(audio_chunk)) if len(audio_chunk) > 0 else 0
            
            # Parametro STEP: definisce quanta strada facciamo verso il target in questo frame.
            step = min(0.01 + (rms * 0.1), 0.2) 

            if mlp_latent_vector is not None:
                # Nuova posizione suggerita dal MLP
                new_target = torch.as_tensor(mlp_latent_vector, device=self.device).float().unsqueeze(0)
                # Normalizzazione sferica
                self.target_z = new_target / (new_target.norm() + 1e-8) * np.sqrt(self.latent_dim)

            # 2. INTERPOLAZIONE SFERICA (Slerp-like)
            # Usiamo lerp per scivolare verso il target in modo controllato
            self.current_z = torch.lerp(self.current_z, self.target_z, step)
            
            # 3. RUMORE DI DRIFT DIMINUITO per togliere il "tremolio" eccessivo.
            self.current_z += torch.randn_like(self.current_z) * 0.003
            
            # 4. GENERAZIONE FP16 (vettore latente -> StyleGAN)
            # force_fp32=False abilita i Tensor Core (velocità raddoppiata su RTX).
            img = self.model(self.current_z, self.c, force_fp32=False, noise_mode='const')
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8) # # POST-PROCESSING: Riordina i canali (H,W,C), scala i colori (0-255) e converte in numeri interi.
            return img[0].cpu().numpy() # GPU -> RAM(CPU)