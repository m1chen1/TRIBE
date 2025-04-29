import numpy as np 
import umap
import matplotlib.pyplot as plt

# Load latent vectors
latent_vectors = np.load('/home/TRIBE_SHARE/Clustering/umap_input_one.npy')
print(f"Loaded {latent_vectors.shape[0]} embeddings of dimension {latent_vectors.shape[1]}.")

# Create the UMAP model
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    verbose=True 
)

print("Starting UMAP dimensionality reduction...")
umap_embeddings = umap_model.fit_transform(latent_vectors)
print("UMAP dimensionality reduction complete.")

# Save embeddings for HDBSCAN
np.save('/home/TRIBE_SHARE/Clustering/umap_embeddings.npy', umap_embeddings)
print("UMAP embeddings saved to /home/TRIBE_SHARE/Clustering/umap_embeddings.npy.")

# Plot and save visualization
plt.figure(figsize=(8, 6))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], alpha=0.5)
plt.title('UMAP projection of the latent vectors')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')

plt.savefig("/home/chen/umap_output.png")
print("UMAP plot saved to /home/chen/umap_output.png.")
