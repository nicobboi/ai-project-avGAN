import os
import sys
import torch
import pickle
import numpy as np
import utils.logutils as log

# IMPORTANTE: Se il pickle richiede moduli specifici di StyleGAN (es. dnnlib),
# assicurati che la cartella stylegan3 sia nel path.
# Se la cartella stylegan3 è nella root del progetto, scommenta la riga sotto:
# sys.path.insert(0, "./stylegan3")

class GANManager:
    def __init__(self, model_path, latent_dim=512, use_gpu=True, eval_mode=True):
        """
        Nota: image_size in StyleGAN3 è definito nel modello stesso, quindi il parametro 
        passato qui potrebbe essere ignorato a favore di G.img_resolution.
        """
        self.model_path = model_path
        # StyleGAN3 di solito usa 512, ma lo leggeremo dinamicamente dal modello
        self.latent_dim = latent_dim 
        
        # LOGICA DI SELEZIONE DEVICE
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

        # Aggiorniamo le dimensioni in base al modello caricato
        if hasattr(self.model, 'z_dim'):
            self.latent_dim = self.model.z_dim
        
        # Gestione Classi (c) per StyleGAN
        # Se il modello non è condizionato, c rimane None
        self.c = None
        if hasattr(self.model, 'c_dim') and self.model.c_dim > 0:
            self.c = torch.zeros([1, self.model.c_dim]).to(self.device)

        # Posizione attuale nello spazio (da dove generiamo l'immagine)
        self.current_z = torch.randn(1, self.latent_dim).to(self.device)
        # Posizione obiettivo verso cui ci stiamo muovendo
        self.target_z = torch.randn(1, self.latent_dim).to(self.device)

    def _load_model(self, eval_mode=True):
        log.info(f"Caricamento modello StyleGAN3 da: {self.model_path}...")

        if not os.path.exists(self.model_path):
            raise FileExistsError(f"Il file del modello non esiste: {self.model_path}")

        try:
            with open(self.model_path, 'rb') as f:
                # Carichiamo il dizionario del network
                network_dict = pickle.load(f)
                
                # Solitamente 'G_ema' è il generatore migliore per inferenza
                if 'G_ema' in network_dict:
                    G = network_dict['G_ema']
                    log.info("Trovato generatore G_ema (Exponential Moving Average).")
                else:
                    G = network_dict['G']
                    log.warning("G_ema non trovato, uso generatore base G.")
                
                G.to(self.device)
                
                # StyleGAN3 è quasi sempre in eval mode per generazione
                G.eval() 
            
            log.success(f"✅ Modello StyleGAN3 caricato! Risoluzione: {G.img_resolution}x{G.img_resolution}")
            return G

        except ModuleNotFoundError as e:
            log.error(f"❌ Errore moduli mancanti: {e}. Assicurati che 'dnnlib' e 'torch_utils' siano visibili a Python.")
            return None
        except Exception as e:
            log.error(f"❌ Errore critico nel caricamento: {e}")
            return None

    def generate_image(self, audio_chunk) -> np.uint8:
        if self.model is None or len(audio_chunk) == 0:
            return None

        with torch.no_grad():
            # 1. FEATURE EXTRACTION DALL'AUDIO
            # Calcoliamo il volume medio del chunk
            volume = np.linalg.norm(audio_chunk) / np.sqrt(len(audio_chunk))
            
            # 2. LOGICA DI NAVIGAZIONE (LATENT WALK)
            base_speed = 0.005 
            dynamic_speed = min(volume * 0.1, 0.5) 
            step = base_speed + dynamic_speed

            # Interpolazione lineare (Lerp)
            self.current_z = (1 - step) * self.current_z + step * self.target_z

            # 3. GESTIONE TARGET
            distance_to_target = torch.norm(self.target_z - self.current_z)
            if distance_to_target < 0.2:
                self.target_z = torch.randn(1, self.latent_dim).to(self.device)

            # 4. GESTIONE IMPULSI ("Kick")
            if volume > 0.5:
                kick_impact = torch.randn(1, self.latent_dim).to(self.device) * volume * 0.2
                self.current_z += kick_impact

            # Normalizzazione sferica (Importante per StyleGAN)
            # StyleGAN si aspetta input normalizzati su una ipersfera
            self.current_z = self.current_z / self.current_z.norm() * np.sqrt(self.latent_dim) # Opzionale su SG3, ma spesso aiuta la stabilità nelle animazioni

            # 5. GENERAZIONE DELL'IMMAGINE
            # StyleGAN3 call: G(z, c, ...)
            # noise_mode='const' evita lo "sfarfallio" della grana pellicola tra i frame
            img = self.model(self.current_z, self.c, force_fp32=True, noise_mode='const')

            # --- Formattazione Output ---
            # Output è [1, 3, H, W] in range [-1, 1]
            
            # 1. Permute da (N, C, H, W) a (N, H, W, C)
            img = img.permute(0, 2, 3, 1)
            
            # 2. Scaling da [-1, 1] a [0, 255]
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            
            # 3. Estrazione array numpy per display/salvataggio
            final_image = img[0].cpu().numpy()
            
            return final_image