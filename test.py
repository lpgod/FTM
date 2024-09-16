import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.patches import Ellipse

# Parameters
n_objects = 5
n_intervals = 5
origin = np.array([0, 0])
destination = np.array([10, 10])
e = 1  # Maximum allowable distance between line segments
MinLns = 2  # Minimum number of neighbors for a core line
# Generate random paths (each row is a path for one object)
paths = np.zeros((n_objects, n_intervals + 1, 2))
for i in range(n_objects):
    paths[i, 0] = origin
    paths[i, -1] = destination
    for t in range(1, n_intervals):
        # Generate intermediate random points
        paths[i, t] = origin + (destination - origin) * t / n_intervals + np.random.randn(2) * 0.5

# # Function to calculate the average Euclidean distance between two line segments
# def line_segment_distance(line1, line2):
#     dists = np.sqrt(np.sum((line1 - line2) ** 2, axis=1))
#     return np.mean(dists)


def divide_line(start, end, L):
    # Start and end are numpy arrays representing the beginning and ending points
    C = 1  # Assume C is some constant, which you can modify as needed
    points = []
    
    for l in range(L + 1):
        sigma = C / (L + 1)
        x_coord = sigma * end[0] + (1 - sigma) * start[0]
        y_coord = sigma * end[1] + (1 - sigma) * start[1]
        points.append((x_coord, y_coord))
    
    return points


def calculate_distances(paths, interval, L):
    distance_matrix = np.zeros((n_objects, n_objects))
    
    for i, j in combinations(range(n_objects), 2):
        start_i, end_i = paths[i, interval], paths[i, interval + 1]
        start_j, end_j = paths[j, interval], paths[j, interval + 1]
        
        # Divide both line segments into L+1 parts
        line_i_points = divide_line(start_i, end_i, L)
        line_j_points = divide_line(start_j, end_j, L)
        
        # Compute the average distance between corresponding points on the lines
        dists = [np.sqrt((x1 - x2)**2 + (y1 - y2)**2) for (x1, y1), (x2, y2) in zip(line_i_points, line_j_points)]
        distance_matrix[i, j] = distance_matrix[j, i] = np.mean(dists)
    return distance_matrix


# Find neighbors of each line segment within distance e
def find_neighbors(dist_matrix, e):
    neighbors = []
    for i in range(n_objects):
        neighbor_list = np.where(dist_matrix[i] <= e)[0]
        neighbors.append(neighbor_list)
    return neighbors

# Cluster line segments based on MinLns
def cluster_segments(neighbors, MinLns):
    clusters = []
    visited = set()

    for i in range(n_objects):
        if i not in visited and len(neighbors[i]) >= MinLns:
            cluster = set(neighbors[i])
            visited.update(cluster)
            clusters.append(cluster)

    return clusters

# Function to calculate the center and size of the ellipse for a cluster
def get_cluster_ellipse(cluster, paths, interval):
    cluster_points = np.array([paths[obj, interval] for obj in cluster] +
                              [paths[obj, interval + 1] for obj in cluster])
    center = np.mean(cluster_points, axis=0)
    cov = np.cov(cluster_points, rowvar=False)

    # Eigenvalues and eigenvectors for the ellipse axes
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    # Width and height of the ellipse
    width, height = 2 * np.sqrt(eigvals)
    return center, width, height, angle

# Plot the clusters with ellipses and lines around them
def plot_clusters(clusters, paths, interval):
    plt.figure(figsize=(8, 8))

    for i, path in enumerate(paths):
        plt.plot(path[:, 0], path[:, 1], marker='o', label=f'Path {i+1}')

    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for i, cluster in enumerate(clusters):
        cluster_color = colors[i % len(colors)]

        # Draw lines around the cluster
        for obj in cluster:
            plt.plot(paths[obj, interval:interval+2, 0], paths[obj, interval:interval+2, 1], color=cluster_color, lw=2)

        # Draw an ellipse around the cluster
        center, width, height, angle = get_cluster_ellipse(cluster, paths, interval)
        ellipse = Ellipse(xy=center, width=width, height=height, angle=angle, edgecolor=cluster_color, facecolor='none', lw=2)
        plt.gca().add_patch(ellipse)

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Clusters at Time Interval {interval}')
    plt.grid(True)
    plt.show()

# Main routine to calculate distances, find neighbors, and plot clusters for each time interval
for interval in range(n_intervals):
    dist_matrix = calculate_distances(paths, interval , 3)
    print(dist_matrix)
    neighbors = find_neighbors(dist_matrix, e)
    clusters = cluster_segments(neighbors, MinLns)
    plot_clusters(clusters, paths, interval)
    
