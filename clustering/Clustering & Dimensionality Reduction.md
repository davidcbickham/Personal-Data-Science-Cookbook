---
tags:
  - clustering
  - dimensionality-reduction
date_created: 2025-09-21
date_modfied: 2025-09-21

---
# **Topics**

- Clustering Algorithms
	- [[#Clustering Algorithm Comparison]]
	- [[#Kmeans Clustering]]
	- [[#Hierarchical Clustering]]
	- [[#DBSCAN Clustering]]
	- [[#HDBSCAN Clustering]]
	- Graph-based Clustering
- Dimensionality Reduction
	- [[#PCA]]
	- [[#UMAP]]

---
# Clustering Algorithm Comparison


|                            | **K-means**                                                                                                                                         | **Agglomerative Hierarchical**                                                                                                                                             | **Density-based (e.g., DBSCAN)**                                                                                                                                                             |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Description**            | Partitions data by iteratively assigning points to nearest centroid and updating centroids until convergence                                        | Builds a tree (dendrogram) of nested clusters by successively merging the closest pairs of clusters from bottom-up                                                         | Identifies clusters as dense regions of points separated by low-density areas, marking sparse points as outliers                                                                             |
| **Determining # Clusters** | ~={purple}Must specify k upfront=~<br>Common methods: elbow method, silhouette score                                                                | ~={purple}Doesn't require pre-specification=~<br>Cut dendrogram at desired height to get clusters                                                                          | ~={purple}Automatically determined by density parameters=~<br>No need to specify number in advance                                                                                           |
| **Pros**                   | ~={purple}Fast and scalable to large datasets=~<br>Simple to understand and implement<br>Works well with spherical, evenly-sized clusters           | ~={purple}Flexible - can explore different granularities=~<br>Captures various cluster shapes<br>~={purple}Produces interpretable dendrogram=~<br>No need to pre-specify k | ~={purple}Discovers arbitrary shapes<br>Robust to outliers and noise=~<br>Automatically finds number of clusters<br>No assumptions about cluster shape                                       |
| **Cons**                   | Assumes spherical clusters<br>~={purple}Sensitive to initialization and outliers=~<br>Requires specifying k<br>Struggles with varying cluster sizes | ~={purple}Computationally expensive (O(n²) to O(n³))<br>Not scalable to very large datasets=~<br>Sensitive to noise and outliers                                           | ~={purple}Requires tuning density parameters (ε, min_points)=~<br>Struggles with varying densities<br>Performance depends heavily on parameter choice<br>Can miss clusters in sparse regions |

<br>

---

# Kmeans Clustering

<br>




> [!summary] Reference Material
> - Clustering Notebooks: https://drive.google.com/drive/folders/1v4Ayn8ZWFZItrIeSsF9lSrRIWPbgKtbp?dmr=1&ec=wgc-drive-hero-goto
> 	- Churn Customer Clustering Demo: [churn_customer_clustering_demo.ipynb](https://colab.research.google.com/drive/1-aXK4AtVKMVfBrs245LR4nIeoc4AQ26Q#scrollTo=a780d4f8)
> 	- Kmeans Yellowbrick Demo: [kmeans_yellowbrick_demo.ipynb](https://colab.research.google.com/drive/1T2N3pN3nknW4TjBzjX3-ibzh2oQoGwGd)

<br>

## Algorithm

- **Initialization** (Kmeans++ better than random)
	- 1. Select centroid randomly
	- 2. Compute distances from this point
	- 3. Distances converted to probability (relative to sum distances from step 2)
	- 4. Choose next centroid that is far away from current centroid
- **Assignment step**: Assign each data point to the nearest centroid.
- **Update step**: Recompute each centroid as the mean of the assigned points.
- **Repeat**: Alternate assignment and update until convergence or max iterations.

<br>

## Scikit-learn Implementation

| Parameter      | Type                                               | Default                                     | Description                                                                                                                                 |
| -------------- | -------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `n_clusters`   | int                                                | 8                                           | Number of clusters to form.                                                                                                                 |
| `init`         | {`'k-means++'`, `'random'`} or ndarray or callable | `'k-means++'`                               | Method for initialization. `'k-means++'` spreads out initial centers, `'random'` picks random samples. Can also pass custom centers.        |
| `n_init`       | int                                                | 10 (scikit-learn ≤1.1) <br> `'auto'` (≥1.2) | Number of times the algorithm will run with different centroid seeds. Best result (lowest inertia) is kept. `'auto'` chooses heuristically. |
| `max_iter`     | int                                                | 300                                         | Maximum number of iterations for a single run.                                                                                              |
| `tol`          | float                                              | 1e-4                                        | Tolerance for convergence (relative change in inertia).                                                                                     |
| `random_state` | int, RandomState instance, or None                 | None                                        | Controls randomness of centroid initialization. Use for reproducibility.                                                                    |
| `algorithm`    | {`'lloyd'`, `'elkan'`}                             | `'lloyd'`                                   | K-Means algorithm variant. `'lloyd'` is standard; `'elkan'` can be faster on well-separated clusters.                                       |
| `copy_x`       | bool                                               | True                                        | Whether to copy data before clustering. If False, input data may be modified.                                                               |
| `verbose`      | int                                                | 0                                           | Verbosity mode (higher = more logging).                                                                                                     |

---

<br>

## Evaluating Cluster Quality

<br>



### Inertia (Distortion)

<br>

> [!tip] Inertia Summary
>- Definition: Sum of squared distances from each point to its cluster centroid
>- Also called: Within-Cluster Sum of Squares (WCSS) or distortion score
>- What it measures: How tightly packed clusters are (cohesion only)
>- Behavior: Always decreases as number of clusters (k) increases
>- Usage: Used in elbow method to find optimal k
>- Limitation: Doesn't consider separation between clusters

Inertia = Σᵢ₌₁ⁿ ||xᵢ - cⱼ||²
Where xᵢ is data point i, cⱼ is the centroid of cluster j that contains point i

<br>

---

### Silhouette

> [!tip] Silhouette Summary
> - The silhouette score is a popular metric for evaluating clustering quality that measures how well-separated and cohesive your clusters are.
> - Better for finding optimal number of clusters
> 	- Cohesion: How close points are to others in their own cluster
> 	- Separation: How far points are from points in other clusters
> 
>- Calculation:
>	- Calculate a(i) - average distance to all other points in the same cluster
>	- Calculate b(i) - average distance to all points in the nearest neighboring cluster
>	- Silhouette score for point i: s(i) = (b(i) - a(i)) / max(a(i), b(i))
> 
> - Analysis Level: Indi~={purple}vidual data points, cluster level (avg.), overall (avg.)=~
> - Pros: Works with any distance metric, algorithm agnostic
> - Cons: Computationally expensive

|Average Silhouette Score|Interpretation|
|---|---|
|**≈ 1.0**|Very strong structure — clusters are well separated and tight.|
|**0.5 – 1.0**|Good structure — clusters are reasonably well defined.|
|**0.25 – 0.5**|Weak structure — clusters may overlap or not be well separated.|
|**0.0 – 0.25**|Very weak or no structure — clustering may not be meaningful.|
|**< 0.0**|Poor clustering — many points assigned to the wrong cluster.|
<br>

**Silhouette Plot**
`yellowbrick`: https://www.scikit-yb.org/en/latest/api/cluster/silhouette.html
`scikit-learn`: manual

![[silhouette_plot.png]]

**Interpretation:**
- Each data point/score is a line, more width = more data points
- Score extends to right (positive) or left (negative)
- Assess avg. score across clusters
- Assess points within clusters with low scores

![[silhoette_score.png]]

<br>

---


## Distance Metrics

<br>

| **Distance Metric** | **Description** | **Best For** | **Formula/Calculation** | **Range** | **Handles Continuous** | **Handles Categorical** | **Scale Sensitive** | **Normalization Needed** | **Sklearn Support** | **Usage Example** |
|---------------------|-----------------|--------------|-------------------------|-----------|------------------------|-------------------------|---------------------|--------------------------|---------------------|-------------------|
| **Euclidean** | Straight-line distance between two points in n-dimensional space | Continuous numerical features with similar scales | √(Σ(x_i - y_i)²) | [0, ∞) | ✅ Yes | ❌ No (requires encoding) | ✅ Yes | ✅ Recommended | ✅ Native: `metric='euclidean'` | `from sklearn.metrics import silhouette_score`<br>`score = silhouette_score(X, labels, metric='euclidean')` |
| **Hamming** | Proportion of positions at which corresponding elements differ | Categorical features, binary data, strings of equal length | (# of differing positions) / (total positions) | [0, 1] | ❌ No (discretization needed) | ✅ Yes | ❌ No | ❌ Not applicable | ✅ Native: `metric='hamming'` | `from sklearn.metrics import silhouette_score`<br>`score = silhouette_score(X_categorical, labels, metric='hamming')` |
| **Gower** | Weighted average of distances for mixed data types; handles continuous, categorical, and ordinal features with different metrics per feature type | Mixed data types (continuous + categorical together) | Weighted average: Σ(w_i × δ_i) / Σw_i<br>where δ_i uses appropriate metric per feature type | [0, 1] | ✅ Yes | ✅ Yes | Partially (built-in normalization) | ❌ Not needed (handles automatically) | ❌ Not native<br>Use `gower` package or precompute with `metric='precomputed'` | `import gower`<br>`from sklearn.metrics import silhouette_score`<br>`distance_matrix = gower.gower_matrix(X_mixed)`<br>`score = silhouette_score(distance_matrix, labels, metric='precomputed')` |

<br>

## Categorical Features

~={purple}**K-Modes (all categorical data)**
=~
- Hamming distance: all nominal categorical data
- Means replaced with modes
- Based of feature category frequencies
- python package: `kmodes`

<br>

~={purple}**K-Prototypes (Mixed data)**=~
- Gower distance: `gower` python package
- Numeric Data: implements something similar to MinMax scaling
- Categorical Data: Hamming distance (% of mismatches)
- python package: `KPrototypes`
- Apply customer distance function using `from sklearn.metrics import pairwise_distances`

<br>

---


## Pros and Cons

<br>



- **PROS:**
	- Computationally efficient
	- Centroid representation
- **CONS:**
	- Sensitive to outliers
	- Struggles detecting clusters of varying sizes
	- Limiting learning of some shapes
	- Evaluation metrics (slow to calculate)


<br>

## Summary

![[Kmeans_summary.png]]

<br>


---

# Hierarchical Clustering


<br>

> [!summary] Reference Material
> - Colab Notebooks: [Click Here](https://drive.google.com/drive/folders/1uUPF35h6HXqrg7fPRbTU6jFzAQKS50Js?dmr=1&ec=wgc-drive-hero-goto)
> 	- [agglomerative_clustering_scipy_demo_1.ipynb](https://colab.research.google.com/drive/1AWe7T-T-BbXDj0NTeuDYNishauERN0ke)
> 	- [agglomerative_clustering_scipy_demo_2.ipynb](https://colab.research.google.com/drive/11lhVoxGiJNWMJ3-T3NzfrB_LOiD0zPgj)
> 	- [agglomerative_clustering_scipy_demo_3_mixed_features.ipynb](https://colab.research.google.com/drive/1WcQLflim8sVmC3OF4Ux535qgdPTHJ8tw)
> - **Overview**
> 	- Distance based algorithm
> 	- Based on dendrogram: tree based linkage diagram of data clusters
> 		- Cophenetic distance: measures dissimilarity between 2 points in dendrogram
> 	- Approaches:
> 		- Top-down: starts with all data points in single group
> 		- Bottom-up (Agglomerative): Begins with individual data points
> 	- Number of clusters depends on where you cut dendrogram
> 	- Use Cases:
> 		- Clustering
> 		- Visual Analysis (via Dendrograms)
> 		- Small Datasets (Visual Analysis focus) -> Computationally inefficient



<br>

## Dendrograms

![[Dendrogram_parts.png]]

<br>

---
<br>

## Linkage

~={purple}**Linkage Matrix**=~
- Dendrogram is described by the linkage matrix
- Each row represents a merging event
- Distance represents "height" of the merge on dendrogram (cophenetic distance)

<br>


~={purple}**Overview of Linkage Methods**=~

| Linkage Method | Description                                               | Distance Calculation                                                     | Characteristics                                                                                                                                         |
| -------------- | --------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Single**     | Nearest neighbor                                          | Minimum distance between any two points in different clusters            | Tends to create long, elongated clusters (chaining effect)<br>Sensitive to outliers and noise<br>Good for non-spherical clusters                        |
| **Complete**   | Farthest neighbor                                         | Maximum distance between any two points in different clusters            | Creates compact, spherical clusters<br>Less sensitive to outliers than single linkage<br>Prefers clusters of similar diameter                           |
| **Average**    | UPGMA (Unweighted Pair Group Method with Arithmetic Mean) | Mean distance between all pairs of points in different clusters          | Balances single and complete linkage<br>More robust to outliers<br>Produces moderate cluster shapes                                                     |
| **Ward**       | Minimum variance                                          | Minimizes within-cluster variance (sum of squared distances to centroid) | Tends to create spherical, evenly-sized clusters<br>Most similar to k-means objective<br>Popular and generally effective<br>Requires Euclidean distance |
<br>

---

<br>

## SciPy Implementation

<br>

![[agglomerative_clustering_scipy.png]]

<br>

---

<br>

## Scikit-learn Implementation

<br>

![[agglomerative_sklearn.png]]

<br>


## Summary

![[Agglomerative_summary.png]]
<br>


---
# DBSCAN Clustering

<br>

> [!summary] Reference Material
> - Google Drive: https://drive.google.com/drive/folders/1GywIIIkqTC03750hWCBRO71bcMYqeh-I?dmr=1&ec=wgc-drive-hero-goto
> - DBSCAN From Scratch: [1_DBSCAN_from_scratch.ipynb](https://colab.research.google.com/drive/17s0e7LmQl4-MV9Si3Y2hR0gkOYZS-qOy)
> 	- Includes `sklearn.neighbors.NearestNeighbors` overview
> - Notebook 2
> - Notebook 3
> <br>
> **Overview**
> - Density-based clustering => radius based clustering
> 	- Points with radius neighborhood of other points clustered together
> 	- `sklearn.neighbors.NearestNeighbors`
> - Data Points Classified as:
> 	- **Core point**: A point with at least min_samples neighbors (including itself) within epsilon distance.
> 	- **Border point**: A point within epsilon distance of a core point but has fewer than min_samples neighbors itself.
> 	- **Noise point**: A point that is neither core nor border (isolated, not within epsilon of any core point).
> - Only 2 Hyperparameters (require tuning)
> 	- Epsilon Radius: radial distance for neighbors
> 	- Min Samples: min number of neighbors to define core point
> - Choice of 3 Nearest Neighbor search algorithms
> 	- **Brute Force:** Computes distance from query point to every single point in the dataset without any optimization.
> 	- **KD-Trees:** Binary tree that partitions space along alternating dimensions, efficient for low-dimensional data but degrades in high dimensions.
> 	- ~={purple}**Ball Tree:**=~ Hierarchical tree structure where each node is a hypersphere containing points, more robust than KD-trees in higher dimensions. (~={purple}Usually best option=~)



<br>

## Nearest Neighbors Scikit-learn

Utilized in DBSCAN from scratch notebook: [1_DBSCAN_from_scratch.ipynb](https://colab.research.google.com/drive/17s0e7LmQl4-MV9Si3Y2hR0gkOYZS-qOy#scrollTo=469a0d5b)

`sklearn.neighbors.NearestNeighbors`

| Parameter       | Description                                 | Common Values                                           |
| --------------- | ------------------------------------------- | ------------------------------------------------------- |
| `n_neighbors`   | Number of neighbors to find                 | 5 (default), depends on task                            |
| `radius`        | Range for radius-based queries              | Used for `radius_neighbors()`                           |
| `algorithm`     | NN search algorithm to use                  | `'auto'`, `'ball_tree'`, `'kd_tree'`, `'brute'`         |
| `leaf_size`     | Leaf size for tree algorithms               | 30 (default), affects speed/memory tradeoff             |
| `metric`        | Distance metric to use                      | `'euclidean'`, `'manhattan'`, `'minkowski'`, `'cosine'` |
| `p`             | Power parameter for Minkowski metric        | 2 (Euclidean), 1 (Manhattan)                            |
| `metric_params` | Additional keyword args for metric function | Dict of parameters                                      |
| `n_jobs`        | Number of parallel jobs                     | -1 (all CPUs), 1 (default)                              |

<br>

<br>


---
<br>

## Scikit-learn Implementation

`sklearn.cluster.DBSCAN`

### DBSCAN Parameters

| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| `eps` | Maximum distance between two samples to be considered neighbors | 0.5 (default), depends on data scale |
| `min_samples` | Minimum number of samples in a neighborhood for a core point | 5 (default), often set to dimensions + 1 |
| `metric` | Distance metric to use | `'euclidean'` (default), `'manhattan'`, `'precomputed'` |
| `metric_params` | Additional keyword arguments for the metric function | Dict of parameters |
| `algorithm` | NN search algorithm to use | `'auto'`, `'ball_tree'`, `'kd_tree'`, `'brute'` |
| `leaf_size` | Leaf size for tree algorithms | 30 (default), affects speed/memory tradeoff |
| `p` | Power parameter for Minkowski metric | 2 (default, Euclidean) |
| `n_jobs` | Number of parallel jobs for neighbor search | None (default), -1 (all CPUs) |

### DBSCAN Attributes

| Attribute | Description | Data Type/Shape |
|-----------|-------------|-----------------|
| `core_sample_indices_` | Indices of core samples | ndarray, shape (n_core_samples,) |
| `components_` | Copy of each core sample found by training | ndarray, shape (n_core_samples, n_features) |
| `labels_` | Cluster labels for each point (-1 for noise) | ndarray, shape (n_samples,) |
| `n_features_in_` | Number of features seen during fit | int |
| `feature_names_in_` | Names of features seen during fit (if input is DataFrame) | ndarray, shape (n_features_in_,) |

### DBSCAN Methods

| Method | Description |
|--------|-------------|
| `fit(X)` | Perform DBSCAN clustering from features |
| `fit_predict(X)` | Perform clustering and return cluster labels |
| `get_params()` | Get parameters for this estimator |
| `set_params(**params)` | Set parameters of this estimator |

<br>


---

## Tuning DBSCAN Parameters

Tuning DBSCAN Parameters: [2_Tuning_DBSCAN_parameters.ipynb](https://colab.research.google.com/drive/1RQ_eC0oMbYvVaCg9st7shmUI76c8BxvT#scrollTo=722b3a46)

- DBSCAN heavily impacted by choice of min samples and radius
- Selecting Min Samples:
	- min_samples > num features + 1
	- min_samples  = 2 * num features (start here)
	- min_samples = ln(total data points)
- Selecting Epsilon Radius:
	- EPS Knee Method



---

## Cluster Quality Validation

<br>


- Metric to measure cluster quality of density-based clustering
	- Core Distances
	- Mutual Reachability Distances & Mutual Reachability Graph
	- Density Separation
	- Density Sparseness
- Minimum Spanning Tree


<br>

### Cluster Validation Coefficient

![[dbcv.png]]
<br>

<br>


## Summary

| Pros | Cons |
|------|------|
| Discovers clusters of arbitrary shapes (not limited to spherical) | Struggles with clusters of varying densities |
| Automatically detects and handles outliers/noise points | Sensitive to parameter selection (eps and min_samples) |
| No need to specify number of clusters beforehand | Difficult to tune parameters without domain knowledge |
| Works well with spatial data and geographic clustering | Poor performance in high-dimensional spaces (curse of dimensionality) |
| Robust to outliers - doesn't force every point into a cluster | Not deterministic with border points (can be assigned to different clusters) |
| Computationally efficient with spatial indexing (O(n log n)) | Requires meaningful distance metric for mixed/categorical data |
| Can find clusters within clusters (nested structures) | Memory intensive for large datasets with distance matrices |
| Based on density, which often aligns with real-world patterns | Results heavily depend on distance metric choice |
| No assumptions about cluster distribution | Difficult to validate - traditional metrics may not apply well |


<br>


---

<br>

# HDBSCAN Clustering


> [!summary] Reference Material
> Google Drive: https://drive.google.com/drive/folders/1GywIIIkqTC03750hWCBRO71bcMYqeh-I?dmr=1&ec=wgc-drive-hero-goto
> [6_HDBSCAN_from_scratch.ipynb](https://colab.research.google.com/drive/1nNhhtjjR5OWiDaGsaeq36cCzVpdgdXZa)
> [9_HDBSCAN_demo_mixed_features.ipynb](https://colab.research.google.com/drive/1jN6o772RnXzXV9dFTUTtFwPVXJiA3FPd)


## Comparison

| Aspect | DBSCAN | HDBSCAN |
|--------|---------|----------|
| **Density handling** | Single, global density | Multiple, varying densities |
| **Parameters** | eps + min_samples | min_cluster_size (+ optional min_samples) |
| **Output** | Flat clustering | Hierarchical clustering |
| **Parameter sensitivity** | High | Lower |
| **Complexity** | Simpler | More complex |
| **Speed** | Faster | Slower |


## Use Cases

| Use DBSCAN when: | Use HDBSCAN when: |
|------------------|-------------------|
| Your clusters have similar densities | Your clusters have varying densities (very common in real data) |
| You need speed and simplicity | You're unsure about parameters |
| You have good domain knowledge for setting `eps` | You want more robust results |
| Working with smaller datasets | You need cluster hierarchy information |
| | Computational cost isn't a primary concern |

<br>




---

# PCA


> [!summary] Reference Material
> Colab Notebooks: https://drive.google.com/drive/folders/18aD4uK5LT_4s-gALdkUrnqEs3pT96k5U?dmr=1&ec=wgc-drive-hero-goto


<br>


## Summary



<br>


![[PCA_summary.png]]


<br>


---

# UMAP


> [!summary] Reference Material
> Colab Notebooks: https://drive.google.com/drive/folders/18aD4uK5LT_4s-gALdkUrnqEs3pT96k5U?dmr=1&ec=wgc-drive-hero-goto


<br>

**UMAP Import in Google Colab**
```python
# umap import in google colab
# Remove all umap-related packages
!pip uninstall umap umap-learn -y

# Clear any cached modules
import sys
modules_to_remove = [key for key in sys.modules.keys() if 'umap' in key]
for module in modules_to_remove:
    del sys.modules[module]

# Reinstall umap-learn
!pip install umap-learn

# Now try importing with the full path
import umap.umap_ as umap
import umap.plot as umap_plot
```

<br>


## Overview

- UMAP (Uniform Manifold Approximation and Projection) is a non-linear dimensionality reduction technique that excels at preserving both local and global structure in data.
- Preserves neighborhoods better than PCA or t-SNE

## Algorithm

1. Step 1: Build local neighborhood graph
	1. For each point, find k nearest neighbors
	2. Uses fuzzy set theory to handle overlapping neighborhoods
2. Step 2: Construct high-dimensional representation
3. Step 3: Find low-dimensional embedding
4. Step 4: Iterative optimization
	1. Use stochastic gradient descent to minimize differences

## Key Advantages

- Better global structure preservation than t-SNE
- Faster computation
- More stable results across runs
- Works well for clustering visualization
- Can handle larger datasets

## Common Use Cases

- 2-D Data visualization
- Preprocessing for downstream ML tasks
- Exploratory data analysis
- Single-cell genomics analysis

## Summary

![[UMAP_summary.png]]

---
