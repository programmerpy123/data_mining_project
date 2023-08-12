from itertools import count

import pandas as pd

from dbscan import DBSCAN
from matplotlib import pyplot as plt
# from evaluation_criteria import davies_bouldin_index
from sklearn.metrics import davies_bouldin_score
def open_csv_file():
    data = []
    with open('datasetnew3.csv', 'r') as f:
        for line in f:
            if line.startswith('grade_point_average'):  # Skip header row
                continue
            fields = line.strip().split(',')
            point = tuple([float(field) for field in fields])
            data.append(point)
    return data

def show_on_plot(labels, data):
        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'brown', 'pink']
        for i in range(len(data)):
            if labels[i] == -1:
                plt.scatter(data[i][0], data[i][1], color='black', s=20)
            else:
                plt.scatter(data[i][0], data[i][1], color=colors[labels[i] % len(colors)], s=20)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('DBSCAN Clustering')
        plt.show()



if __name__ == "__main__":
    data = open_csv_file()
    labels, cluster_count = DBSCAN(data, eps=0.09, min_samples=8).DB
    print(f"clusters count: ",{cluster_count})
    X = pd.read_csv('datasetnew3.csv')
    print(f"davis: ",davies_bouldin_score(X,labels))
    counter = 0
    for i in labels:
        if i == -1:
            counter+=1
    print("count of outlier is: ",counter)


    show_on_plot(labels,data)
    # show_on_plot(labels, data)

