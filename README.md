# SeismoQNet
Hybrid Quantum–Deep Learning Pipeline for Seismic Signal Detection

SeismoQNet is a research-oriented project that explores the use of **Quantum Machine Learning (QML)** for detecting seismic signals in noisy environments. The system combines **classical signal processing, deep learning, and quantum clustering** to identify earthquake P-wave arrivals from seismic data.

This work is inspired by the research paper:

**Dalai & Kumar (2025)**  
*Integrating quantum clustering with unsupervised deep learning for first arrival picking in local seismic events*  
Published in *Geophysical Journal International*.

---

# Project Objective

Seismic sensors often operate in **high-noise environments**, especially in urban areas. Traditional algorithms such as **STA/LTA triggers** may fail when background vibrations from traffic, construction, or environmental noise are present.

This project explores whether **quantum-enhanced machine learning** can improve the detection of seismic signals by separating earthquake patterns from noise more effectively.

---

# System Architecture

The system follows a **Hybrid Quantum–Deep Learning Pipeline** consisting of four phases:

### Phase 1 — Signal Transformation
Raw seismic signals are converted into a time–frequency representation.

- Input: Raw seismic waveform
- Method: **Generalized S-Transform (GST)**
- Output: **Spectrogram**

The spectrogram allows the system to observe frequency changes over time, which is useful for identifying P-wave patterns.

---

### Phase 2 — Feature Extraction
The spectrogram contains high-dimensional data, which must be compressed before being processed by quantum circuits.

- Method: **Convolutional Autoencoder (CAE)**
- Process:
  - Encoder compresses spectrogram
  - Latent vector is extracted

Additional classical features are added:

- STA/LTA ratio
- Mean
- Variance
- Crest factor

Output: **Enriched Feature Vector**

---

### Phase 3 — Quantum Classification
The enriched feature vector is processed using quantum machine learning.

Steps:

1. Encode features into quantum states using **Amplitude Encoding**
2. Run **Quantum k-means clustering**
3. Compute similarity using a **Swap Test Circuit**

Output:

- Cluster 0 → Background Noise
- Cluster 1 → Seismic Signal

---

### Phase 4 — Arrival Picking
Once signal segments are identified, the system determines the **exact P-wave arrival time**.

Method:

- Dynamic threshold detection
- Backward verification to locate the first signal onset

Output: **Detected P-wave arrival time**

---

# Technology Stack

### Classical Computing
- Python
- NumPy
- SciPy
- Matplotlib
- ObsPy

### Deep Learning
- PyTorch / TensorFlow
- Convolutional Autoencoders

### Quantum Computing
- Qiskit
- Quantum Circuits
- Swap Test
- Quantum k-means

---

# Dataset Sources

The project will experiment with multiple datasets:

1. **Synthetic Seismic Data**
   - Ricker wavelet simulation with adjustable noise levels

2. **STEAD Dataset**
   - Stanford Earthquake Dataset

3. **Indian Seismic Data**
   - National Center for Seismology (NCS)
   - IRIS seismic database

---

# Project Pipeline

Raw Seismogram  
↓  
GST Transformation  
↓  
Spectrogram  
↓  
Convolutional Autoencoder  
↓  
Latent Feature Vector  
↓  
Feature Enrichment (STA/LTA + statistics)  
↓  
Quantum Encoding  
↓  
Quantum Clustering (q-means)  
↓  
Signal vs Noise Classification  
↓  
P-wave Arrival Detection

---

# Research Goals

- Compare **Classical SVM vs Quantum SVM / Quantum Clustering**
- Test model performance under **different noise levels**
- Evaluate whether quantum feature spaces improve **signal detection accuracy**

---

# Future Work

- Testing on real seismic stations in India
- Running circuits on IBM Quantum hardware
- Extending the model for **S-wave detection**
- Real-time early warning system integration

---

# License
This project is released under the **MIT License**.
