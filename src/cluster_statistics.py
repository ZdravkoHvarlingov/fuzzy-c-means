import matplotlib.pyplot as plt
import numpy as np


class ClusterStatisticsInformator:

    @staticmethod
    def show_statistics(data, centroids, assignments):

        for cluster_num, centroid in enumerate(centroids):
            ClusterStatisticsInformator._show_cluster_statistics(data, cluster_num, centroid, assignments)

        max_assignment = np.max(assignments, axis=1)
        plt.scatter(list(range(0, data.shape[0])), max_assignment, s=1)
        plt.show()

    @staticmethod
    def _show_cluster_statistics(data, cluster_num, centroid, assignments):
        assignments_sorted = assignments[:, cluster_num].argsort()
        cluster_points = data[assignments_sorted]
        cluster_assignments = assignments[assignments_sorted]

        np.set_printoptions(suppress=True, precision=3)
        print(f'############################################# Cluster {cluster_num}')
        print(cluster_points[-10: cluster_points.shape[0], :])
        print(cluster_assignments[-10: cluster_assignments.shape[0], :])
        print(centroid)

        plt.scatter(list(range(0, data.shape[0])), assignments[:, cluster_num], s=1)
        plt.show()

    @staticmethod
    def save_assignments_to_file(assignments):
        with open('results.out', 'w') as out_file:
            for element_assignment in assignments:
                out_file.write(','.join([str(asign) for asign in element_assignment]) + '\n')