# 🦋 The Visual Specimen Vault: 3D Latent Space Explorer

**A deep-learning-powered visual search engine and interactive morphological clustering dashboard.**

Standard image classification models output a single top-1 accuracy prediction, which acts as a "black box." The Visual Specimen Vault breaks open that box. By extracting high-dimensional feature vectors from a pre-trained Convolutional Neural Network and mapping them into a 3D topological space, this application allows users to visually explore how a machine learning model understands the morphological traits of 40 different butterfly species.

## ✨ Core Features
* **Latent Space Clustering:** Squashes 2048-dimensional visual features down to 3 dimensions using UMAP, grouping species by their visual and evolutionary similarities.
* **Interactive 3D Dashboard:** Built with Streamlit and Plotly, allowing users to rotate, zoom, and filter the latent space in real-time.
* **Dynamic Specimen Viewer:** Selecting a cluster in the 3D space dynamically retrieves and displays biological reference images of that species.
* **Live Inference Engine:** Upload a new image of a butterfly to run it through the ResNet50 model in real-time, calculate its Euclidean distance across the 2048-dimensional space, and find its closest visual match in the vault.

## 🛠️ Tech Stack
* **Deep Learning / Computer Vision:** PyTorch, Torchvision (ResNet50)
* **Dimensionality Reduction:** UMAP (`umap-learn`)
* **Frontend Application:** Streamlit
* **Data Visualization:** Plotly 3D
* **Data Manipulation:** NumPy, Pandas, Pillow

## 🚀 How to Run the Pipeline Locally

Because the image dataset and extracted tensors are too large for standard GitHub hosting, you must generate the local data pipeline before launching the application.

### 1. Environment Setup
Clone this repository and install the required dependencies:
```bash
git clone [https://github.com/xuannn0212/Visual-Specimen-Vault.git](https://github.com/YOUR-USERNAME/Visual-Specimen-Vault.git)
cd Visual-Specimen-Vault
python -m venv venv
source venv/bin/activate  # Or .\venv\Scripts\activate on Windows
pip install -r requirements.txt