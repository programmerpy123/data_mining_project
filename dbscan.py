import math

from matplotlib import pyplot as plt


def euclidean_distance(point1, point2):

    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

def get_neighbors(data, point_index, eps):

    neighbors = []
    for i in range(len(data)):
        if euclidean_distance(data[point_index], data[i]) <= eps:
            neighbors.append(i)
    return neighbors

def dbscan(data, eps, min_samples):

    labels = [-1] * len(data)  # Initialize all points as noise (-1)
    cluster_id = 0
    for i in range(len(data)):
        if labels[i] != -1:  # Skip points that have already been assigned to a cluster
            continue
        neighbors = get_neighbors(data, i, eps)
        if len(neighbors) < min_samples:  # Assign as noise if not enough neighbors
            labels[i] = -1
            continue
        cluster_id += 1
        labels[i] = cluster_id
        j = 0
        while j < len(neighbors):
            neighbor_index = neighbors[j]
            if labels[neighbor_index] == -1:  # Add unassigned neighbors to current cluster
                labels[neighbor_index] = cluster_id
                neighbor_neighbors = get_neighbors(data, neighbor_index, eps)
                if len(neighbor_neighbors) >= min_samples:
                     neighbors += neighbor_neighbors
            j += 1
    return labels, cluster_id


data = []
with open('datasetnew3.csv', 'r') as f:
    for line in f:
        if line.startswith('grade_point_average'):  # Skip header row
            continue
        fields = line.strip().split(',')
        point = tuple([float(field) for field in fields[4:8]])
        data.append(point)


labels, cluster_count = dbscan(data, eps=0.04, min_samples=1)

colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'brown', 'pink']
for i in range(len(data)):
    if labels[i] == -1:
        plt.scatter(data[i][0], data[i][1], color='black', s=5)
    else:
        plt.scatter(data[i][0], data[i][1], color=colors[labels[i] % len(colors)], s=5)

plt.xlabel('x')
plt.ylabel('y')
plt.title('DBSCAN Clustering')
plt.show()