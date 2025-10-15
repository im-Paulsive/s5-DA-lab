import numpy as np

# --- Step 1: Take points from the user ---
points = []
n = int(input("Enter number of data points: "))
for i in range(n):
    x, y = map(float, input("Enter x and y for point " + str(i+1) + " separated by space: ").split())
    points.append([x, y])

X = np.array(points)

# --- Step 2: Take initial centroids from the user ---
k = int(input("Enter number of clusters: "))
centroids = []
print("Enter initial centroids:")
for i in range(k):
    cx, cy = map(float, input("Centroid " + str(i+1) + " (x y): ").split())
    centroids.append([cx, cy])

centroids = np.array(centroids)

# --- Step 3: K-Means Algorithm ---
def kmeans(X, centroids, max_iters=100, tol=1e-4):
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(len(centroids))])
        
        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    
    return centroids, clusters

# --- Step 4: Run K-Means ---
final_centroids, clusters = kmeans(X, centroids)

# --- Step 5: Display Results ---
print("\nFinal Centroids:")
for c in final_centroids:
    print(c)

print("\nCluster assignments:")
for i, cluster in enumerate(clusters):
    print("Point", X[i], "-> Cluster", cluster)
