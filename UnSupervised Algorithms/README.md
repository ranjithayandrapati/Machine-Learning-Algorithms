# 🧠 Unsupervised Learning Algorithms

Unsupervised learning is a branch of machine learning where the model learns **patterns and structures from unlabeled data**. Unlike supervised learning, we do not provide explicit input-output pairs — the algorithm discovers hidden structures, groupings, or representations by itself.

---

## 📚 Key Categories

### 1. **Clustering**
Groups data points into similar clusters based on feature similarity.

- **k-Means Clustering** – partitions data into *k* groups by minimizing intra-cluster variance.
- **Hierarchical Clustering** – builds a tree (dendrogram) of clusters, can be agglomerative (bottom-up) or divisive (top-down).
- **DBSCAN (Density-Based Spatial Clustering)** – groups points that are closely packed; identifies noise/outliers.
- **Gaussian Mixture Models (GMM)** – probabilistic clustering using a mixture of Gaussian distributions.

---

### 2. **Dimensionality Reduction**
Reduces the number of features while preserving essential structure.

- **PCA (Principal Component Analysis)** – projects data into directions of maximum variance.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)** – creates a 2D/3D embedding for visualization.
- **UMAP (Uniform Manifold Approximation and Projection)** – preserves local and global structure better than t-SNE.
- **Autoencoders** – neural networks that compress data into a lower-dimensional latent representation.

---

### 3. **Association Rule Learning**
Discovers interesting relationships between variables.

- **Apriori Algorithm** – identifies frequent itemsets and association rules.
- **FP-Growth** – a faster alternative to Apriori for frequent pattern mining.

---

### 4. **Anomaly / Novelty Detection**
Identifies data points that deviate significantly from the norm.

- **Isolation Forest** – isolates anomalies by random partitioning.
- **One-Class SVM** – learns a decision boundary around normal data.
- **Local Outlier Factor (LOF)** – detects density-based anomalies.

---

## 🎯 Common Use Cases
- **Customer Segmentation** (clustering)
- **Market Basket Analysis** (association rules)
- **Dimensionality Reduction** for visualization or preprocessing
- **Anomaly Detection** in fraud detection, cybersecurity, or manufacturing
- **Representation Learning** with autoencoders

---



