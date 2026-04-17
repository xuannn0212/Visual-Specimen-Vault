import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models

# --- 1. Page Configuration ---
st.set_page_config(page_title="The Visual Specimen Vault", layout="wide", page_icon="🦋")

st.title("🦋 The Visual Specimen Vault")
st.markdown("### Interactive Latent Space Explorer & Identifier")
st.write("Upload a butterfly image to find its closest visual matches in the vault using deep learning.")

# --- 2. Data & Model Loading (Cached for speed) ---
@st.cache_data
def load_data():
    # Update these three lines to point to the new full files
    embeddings_3d = np.load('processed/embeddings_3d_full.npy')
    embeddings_high_d = np.load('processed/embeddings_full.npy') 
    labels = np.load('processed/labels_full.npy')
    class_mapping = np.load('processed/class_mapping.npy', allow_pickle=True).item()
    
    species_names = [class_mapping[label] for label in labels]
    
    df = pd.DataFrame({
        'x': embeddings_3d[:, 0],
        'y': embeddings_3d[:, 1],
        'z': embeddings_3d[:, 2],
        'Species': species_names,
        'Original_Index': range(len(species_names))
    })
    return df, class_mapping, embeddings_high_d

# Use cache_resource for PyTorch models so it only loads into memory once
@st.cache_resource
def load_model():
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess

df, class_mapping, embeddings_high_d = load_data()
model, preprocess = load_model()

# --- 3. Sidebar Architecture ---
st.sidebar.header("Vault Controls")
st.sidebar.markdown("---")

# Feature: Image Uploader for Live Inference
uploaded_file = st.sidebar.file_uploader("Upload a Butterfly Image to Identify:", type=["jpg", "jpeg", "png"])

identified_species = None

if uploaded_file is not None:
    st.sidebar.markdown("### Uploaded Specimen")
    img = Image.open(uploaded_file).convert('RGB')
    st.sidebar.image(img, use_container_width=True)
    
    # We add 'expanded=True' and save it as a variable named 'status'
    with st.sidebar.status("Analyzing morphological features...", expanded=True) as status:
        
        st.write("⏳ Preprocessing image to 224x224 tensor...")
        input_tensor = preprocess(img).unsqueeze(0) 
        
        st.write("🧠 Extracting 2048-D features via ResNet50...")
        with torch.no_grad():
            new_features = model(input_tensor).numpy()
        
        st.write("📐 Calculating distances across latent space...")
        distances = np.linalg.norm(embeddings_high_d - new_features, axis=1)
        
        # Find the closest match
        closest_index = np.argmin(distances)
        identified_species = df.iloc[closest_index]['Species']
        
        # Change the title to green when finished and collapse it
        status.update(label="Analysis Complete!", state="complete", expanded=False)
        
    st.sidebar.success(f"**Closest Match:** {identified_species}")
    st.sidebar.markdown("---")

# Feature: Species Selector
selected_species = st.sidebar.multiselect(
    "Highlight Specific Species:",
    options=sorted(df['Species'].unique()),
    default=[identified_species] if identified_species else []
)

# --- Dynamic Specimen Viewer ---
if selected_species:
    st.sidebar.markdown("### Specimen Reference")
    for species in selected_species:
        # Notice we are using 'reference_images' here for the cloud deployment!
        species_path = os.path.join('reference_images', species)
        
        if os.path.exists(species_path):
            images = os.listdir(species_path)
            if images:
                first_image_path = os.path.join(species_path, images[0])
                try:
                    img = Image.open(first_image_path)
                    st.sidebar.image(img, caption=f"Sample: {species}", use_container_width=True)
                except Exception as e:
                    pass

st.sidebar.markdown("---")
st.sidebar.write(f"**Total Specimens Mapped:** {len(df)}")

# --- 4. The 3D Visualization Pipeline ---
# Filter data dynamically based on user selection
if selected_species:
    plot_df = df[df['Species'].isin(selected_species)]
else:
    plot_df = df

# Build the Plotly figure
fig = px.scatter_3d(
    plot_df, x='x', y='y', z='z',
    color='Species',
    opacity=0.8,
    hover_name='Species',
    height=750
)

# Polish the visual elements
fig.update_traces(marker=dict(size=5, line=dict(width=0)))
fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        xaxis_title="Latent Dimension 1",
        yaxis_title="Latent Dimension 2",
        zaxis_title="Latent Dimension 3",
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False)
    )
)

st.plotly_chart(fig, use_container_width=True)