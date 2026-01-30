import torch
import os
import sys
import pickle
import numpy as np
import utils.logutils as log

class GANManager:
    def __init__(self, model_path, latent_dim=512, use_gpu=True):
        self.model_path = model_path
        self.latent_dim = latent_dim 
        
        # SELEZIONE DEVICE
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            log.info("GAN Manager: Modalità GPU (CUDA) attivata.")
        else:
            self.device = torch.device('cpu')
            if use_gpu and not torch.cuda.is_available():
                log.warning("GAN Manager: GPU richiesta ma non trovata. Fallback su CPU.")
            else:
                log.info("GAN Manager: Modalità CPU forzata.")

        # Carica il modello
        self.model = self._load_model()

        if hasattr(self.model, 'z_dim'):
            self.latent_dim = self.model.z_dim
        
        self.c = None
        if hasattr(self.model, 'c_dim') and self.model.c_dim > 0:
            self.c = torch.zeros([1, self.model.c_dim]).to(self.device)

        # Inizializza lo stato
        self.reset_state()

    def _load_model(self, eval_mode=True):
        log.info(f"Caricamento modello StyleGAN3 da: {self.model_path}...")

        if not os.path.exists(self.model_path):
            raise FileExistsError(f"Il file del modello non esiste: {self.model_path}")

        try:
            with open(self.model_path, 'rb') as f:
                network_dict = pickle.load(f)
                if 'G_ema' in network_dict:
                    G = network_dict['G_ema']
                else:
                    G = network_dict['G']
                
                G.to(self.device)
                G.eval() 
            
            log.success(f"✅ Modello StyleGAN3 caricato! Risoluzione: {G.img_resolution}x{G.img_resolution}")
            return G

        except Exception as e:
            log.error(f"❌ Errore critico nel caricamento: {e}")
            return None

    def reset_state(self):
        """Reinizializza i vettori latenti a valori casuali (Reset)"""
        log.info("🔄 GAN Reset: Rigenerazione vettori latenti...")
        self.current_z = torch.randn(1, self.latent_dim).to(self.device)
        self.current_z = self.current_z / self.current_z.norm() * np.sqrt(self.latent_dim)
        
        # Anche il target viene resettato per evitare scatti verso vecchi target
        self.target_z = torch.randn(1, self.latent_dim).to(self.device)

    def get_distance_to_target(self):
        """Restituisce quanto siamo lontani dal target (indicatore di movimento)"""
        if self.target_z is None or self.current_z is None:
            return 0.0
        return torch.norm(self.target_z - self.current_z).item()

    def generate_image(self, audio_chunk, features=None, mlp_latent_vector=None) -> np.uint8:
        """
        Genera frame con movimento fluido verso il target MLP.
        """
        if self.model is None or len(audio_chunk) == 0:
            return None

        with torch.no_grad():
            # 1. CALCOLO VELOCITÀ
            rms = np.linalg.norm(audio_chunk) / np.sqrt(len(audio_chunk))
            base_speed = 0.02           
            dynamic_speed = rms * 0.2   
            step = min(base_speed + dynamic_speed, 0.8)

            # 2. GESTIONE TARGET
            if mlp_latent_vector is not None:
                mlp_target = torch.tensor(mlp_latent_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
                mlp_target = mlp_target / mlp_target.norm() * np.sqrt(self.latent_dim)
                self.target_z = mlp_target
            else:
                # Random walk se silenzio
                distance = torch.norm(self.target_z - self.current_z)
                if distance < 0.2:
                    self.target_z = torch.randn(1, self.latent_dim).to(self.device)

            # 3. INTERPOLAZIONE
            self.current_z = (1 - step) * self.current_z + step * self.target_z

            # 4. RUMORE DI MOVIMENTO (Drift)
            drift_noise = torch.randn(1, self.latent_dim, device=self.device) * 0.05
            self.current_z += drift_noise

            # 5. GENERAZIONE
            self.current_z = self.current_z / self.current_z.norm() * np.sqrt(self.latent_dim)
            img = self.model(self.current_z, self.c, force_fp32=True, noise_mode='const')

            img = img.permute(0, 2, 3, 1)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            final_image = img[0].cpu().numpy()
            
            return final_image