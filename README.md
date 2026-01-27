# 🧠 Real-Time Neural AV

> **A Real-Time Generative Audio-Visual Performance system powered by StyleGAN, Python, and TouchDesigner.**

## 📖 Overview
This application creates real-time, audio-reactive visuals using a deep learning StyleGAN model. It analyzes audio chunks on the fly, generates corresponding latent representations, and renders the visual frames. The output is displayed in a PyQt6 GUI and simultaneously streamed at zero-latency to **TouchDesigner** via the **SpoutGL** protocol for live mapping and post-processing.

## ⚙️ Configuration (Custom Variables)
Before running the application, you can configure the behavior of the visualizer by editing the `### CUSTOM VARIABLES ###` section in `main.py`:

| Variable | Description |
|----------|-------------|
| `FRAMERATE` | The target refresh rate of the GUI and Spout stream (e.g., `FPS.FPS_30`). |
| `SAMPLE_WINDOW_SIZE` | The size of the audio chunk analyzed per frame (e.g., `SampleWindowSize.WS_1024`). Controls the reactivity. |
| `MODEL_PATH` | Path to the PyTorch StyleGAN model `.pt` file. |
| `USE_GPU` | **Boolean (`True`/`False`)**. Enables CUDA acceleration. *Highly recommended for real-time performance and TouchDesigner integration.* |
| `EVAL_MODE` | **Boolean (`True`/`False`)**. Sets the PyTorch model to evaluation mode (disables dropouts/batch norms). |

*Note: The Spout transmission channel is hardcoded as `GAN_Visualizer_TD`.*

---

## ⚙️ Setup & Installation

Follow these steps to configure the development environment and run the application.

### 1. Prerequisites
* **Python 3.14** installed.
* In progress...

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
**4. Run the application:**
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