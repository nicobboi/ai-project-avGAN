# 🧠 Real-Time Neural AV (StyleGAN3 + MLP Audio-Reactive)

> **A Real-Time Generative Audio-Visual Performance system. It uses a Feed-Forward Neural Network (MLP) to bridge audio features into the latent space of a pre-trained StyleGAN3 model.**

## 📖 Overview
This application creates real-time, audio-reactive visuals by chaining two neural networks:
1.  **Audio Feature Extractor**: Analyzes audio chunks using `librosa` to extract spectral contrast, flatness, onset strength, and more.
2.  **Mood Predictor (MLP)**: A lightweight neural network maps these audio features to a high-dimensional Latent Vector ($z$).
3.  **StyleGAN Generator**: Generates visual frames from the latent vectors.

The output is displayed in a PyQt6 GUI and simultaneously streamed at zero-latency to **TouchDesigner** via the **SpoutGL** protocol.

## 📂 Project Structure & Resources
Ensure your project folder has a `resources/` directory containing your trained models:
* `./resources/gan_model.pkl`: The StyleGAN3 pre-trained network.
* `./resources/mood_mlp.pth`: The trained MLP weights for audio-to-latent mapping.
* `./resources/scaler.pkl`: The Scikit-learn scaler for normalizing audio inputs.

## ⚙️ Configuration
You can tweak the system behavior by editing the **Environment Variables** section at the top of `main.py`:

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_GPU` | Enables CUDA acceleration. Required for real-time inference. | `True` |
| `GAN_MODEL_PATH` | Path to the StyleGAN `.pkl` snapshot. | `./resources/gan_model.pkl` |
| `MLP_MODEL_PATH` | Path to the MLP `.pth` weights. | `./resources/mood_mlp.pth` |
| `SPOUT_SENDER_NAME` | The Spout sender name visible in TouchDesigner. | `"GAN_Visualizer_TD"` |

---

## ⚙️ Setup & Installation

### 1. Prerequisites
* **Python 3.12.10**
* **NVIDIA GPU** with **CUDA 12.6** drivers installed.
* Microsoft Visual C++ Redistributable 2022 (x64)

### 2. Automatic Configuration (Venv)
Open your terminal in the project folder and run the following commands in sequence:

**1. Create the virtual environment:**
```bash
python -m venv ./venv
```
**2. Activate the virtual environment:**
- Windows
```bash
.\venv\Scripts\activate
```
- macOS/Linux:
```bash
source venv/bin/activate
```
**3. Install dependecies:**
```bash
python -m pip install -r requirements.txt
```
**4. Run the application.**<br>
**IMPORTANT: Open from "x64 Native Tools Command Prompt for VS 2022**
```bash
python src\main.py
```

### 3. TouchDesigner Setup (⚠️ Requires `USE_GPU = True`)
To achieve smooth real-time performance and send textures to TouchDesigner, ensure you have a dedicated GPU that supports CUDA and set `USE_GPU = True` in `main.py`.

💡 **Quick Start:** A default TouchDesigner project file (`.toe`) is included in this repository.

**Manual Setup (for existing projects):**
1. Run the Python application first so the Spout channel becomes active.
2. Open **TouchDesigner**.
3. Press `TAB` to open the OP Create Dialog and select the **TOP** family.
4. Place a **Spout In TOP** node in your network.
5. In the node parameters (top right), set the **Sender Name** to `GAN_Visualizer_TD`.
6. Connect the output to a **Null TOP** and then to an **Out TOP** to integrate the neural visuals into your TouchDesigner pipeline.

---

## 📄 Usage

1. From your x64 terminal (ensure the virtual environment is active), launch the App: The GUI will open showing "Waiting for audio...".

2. Load Audio: Click "Open" and select a .wav file. The system will take a moment to decode the audio and pre-compute dynamic ranges.

3. Audio will start playing automatically. The MLP will predict visuals based on the audio features in real-time.

4. TouchDesigner: Open TouchDesigner to receive the feed.