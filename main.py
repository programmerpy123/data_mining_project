from itertools import count

import pandas as pd
from sympy.physics.units import cm

from dbscan import DBSCAN
from matplotlib import pyplot as plt
# from evaluation_criteria import davies_bouldin_index
from sklearn.metrics import davies_bouldin_score
def open_csv_file():
    data = []
    with open('datasetnew2.csv', 'r') as f:
        for line in f:
            if line.startswith('norm_year'):  # Skip header row
                continue
            fields = line.strip().split(',')
            point = tuple([float(field) for field in fields])
            data.append(point)
    return data




def show_on_plot(labels, data, counter):
    num_colors = len(labels)  # Number of colors based on the number of labels
    cmap = cm.get_cmap('rainbow', num_colors)  # Create a color map with a rainbow spectrum

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Use '3d' projection for a 3D plot

    for i in range(len(data)):
        if labels[i] == -1:
            ax.scatter(data[i][0], data[i][1], data[i][2], color='black', s=20)
        else:
            color = cmap(i)  # Get the color from the color map
            ax.scatter(data[i][0], data[i][1], data[i][2], color=color, s=20)

    plt.show()



if __name__ == "__main__":
    data = open_csv_file()
    labels, cluster_count = DBSCAN(data, eps=0.1, min_samples=1).DB
    print(f"clusters count: ",{cluster_count})
    X = pd.read_csv('datasetnew2.csv')
    print(f"davis: ",davies_bouldin_score(X,labels))
    counter = 0
    for i in labels:
        if i == -1:
            counter+=1
    print("count of outlier is: ",counter)


    show_on_plot(labels,data,counter)
    # show_on_plot(labels, data)

