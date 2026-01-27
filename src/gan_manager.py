import os
import torch
import numpy as np
from utils.lightweight_gan_cust import Generator
import utils.logutils as log


class GANManager:
    # Aggiungiamo il parametro use_gpu=True come default
    def __init__(self, model_path, image_size=256, latent_dim=256, use_gpu=True, eval_mode=True):
        self.model_path = model_path
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # LOGICA DI SELEZIONE DEVICE
        # Usa CUDA solo se l'utente lo vuole E se è disponibile
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
        self.model = self._load_model(eval_mode=eval_mode)

        # Posizione attuale nello spazio (da dove generiamo l'immagine)
        self.current_z = torch.randn(1, self.latent_dim).to(self.device)
        # Posizione obiettivo verso cui ci stiamo muovendo
        self.target_z = torch.randn(1, self.latent_dim).to(self.device)

    def _load_model(self, eval_mode=True):
        # ... (Il resto del codice rimane uguale, userà self.device automaticamente) ...
        # Copia pure il metodo _load_model e generate_image dal messaggio precedente
        # Assicurati solo che _load_model usi self.device come faceva prima
        log.info("Caricamento modello in corso...")
        
        model = Generator(
            image_size=self.image_size, 
            latent_dim=self.latent_dim
        )

        if not os.path.exists(self.model_path):
            raise FileExistsError(f"Il file del modello non esiste: {self.model_path}")
            

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            gan_weights = checkpoint['GAN']

            gen_weights = {}
            found_ema = False
            
            for k, v in gan_weights.items():
                if k.startswith('GE.'):
                    gen_weights[k.replace('GE.', '')] = v
                    found_ema = True
            
            if not found_ema:
                log.warning("Pesi EMA non trovati, uso pesi Base (G)...")
                for k, v in gan_weights.items():
                    if k.startswith('G.') and not k.startswith('GE.') and 'scaler' not in k:
                        gen_weights[k.replace('G.', '')] = v
            else:
                log.info("Pesi EMA (GE) trovati e caricati.")

            model.load_state_dict(gen_weights, strict=False)
            model.to(self.device)

            if eval_mode:
                model.eval()
            else:
                model.train() 
            
            log.success("✅ Modello caricato e pronto!")
            return model

        except Exception as e:
            log.error(f"❌ Errore critico nel caricamento: {e}")
            return model

    def generate_image(self, audio_chunk) -> np.uint8:
        if self.model is None or len(audio_chunk) == 0:
            return None

        with torch.no_grad():
            # 1. FEATURE EXTRACTION DALL'AUDIO
            # Calcoliamo il volume medio del chunk
            volume = np.linalg.norm(audio_chunk) / np.sqrt(len(audio_chunk))
            
            # 2. LOGICA DI NAVIGAZIONE (LATENT WALK)
            # La velocità di movimento dipende dal volume.
            # - base_speed: velocità minima anche in silenzio (morphing lento continuo)
            # - dynamic_speed: picco di velocità dato dalla musica
            base_speed = 0.005 
            dynamic_speed = min(volume * 0.1, 0.5) # Limita il passo massimo
            step = base_speed + dynamic_speed

            # Interpolazione lineare (Lerp) dal punto attuale verso il target
            self.current_z = (1 - step) * self.current_z + step * self.target_z

            # 3. GESTIONE TARGET (Cambio di direzione)
            # Se siamo arrivati vicini al target, ne scegliamo uno nuovo a caso
            distance_to_target = torch.norm(self.target_z - self.current_z)
            if distance_to_target < 0.2:
                self.target_z = torch.randn(1, self.latent_dim).to(self.device)

            # 4. GESTIONE IMPULSI (Opzionale per effetto "Kick/Cassa")
            # Se c'è un forte picco audio, aggiungiamo un rumore istantaneo
            # che deforma l'immagine sul colpo di batteria
            if volume > 0.5:
                kick_impact = torch.randn(1, self.latent_dim).to(self.device) * volume * 0.2
                self.current_z += kick_impact

            # Normalizzazione (Best practice per non far degradare l'immagine delle GAN nel tempo)
            self.current_z = self.current_z / self.current_z.norm() * np.sqrt(self.latent_dim)

            # 5. GENERAZIONE DELL'IMMAGINE
            generated_tensor = self.model(self.current_z)

            # --- Formattazione Output (Rimane identica) ---
            if torch.isnan(generated_tensor).any():
                generated_tensor = torch.nan_to_num(generated_tensor, nan=0.0)

            img_data = (generated_tensor.clamp(-1, 1) + 1) / 2
            img_data = (img_data * 255).byte()
            final_image = img_data[0].permute(1, 2, 0).cpu().numpy()
            
            return final_image