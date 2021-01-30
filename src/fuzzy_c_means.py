import numpy as np
from numpy.core.fromnumeric import nonzero
from numpy.core.numeric import indices


class FuzzyCMeans:

    NORMALIZED_MAX = 1
    FUZZINESS_COEFFICIENT = 2
    EPSILON_THRESHOLD = 0.005

    def __init__(self, n_clusters):
        self._n_clusters = n_clusters

    def fit(self, data):
        data = self._normalize_data(data)
        centroids = self._randomize_centroids(data)
        assignments = self._get_cluster_assignments(data, centroids)
        change_occured = True

        iteration = 0
        while change_occured:
            print(f'Iteration: {iteration}.')
            centroids = self._calculate_mean_centroids(data, assignments)
            new_assignments = self._get_cluster_assignments(data, centroids)
            change_occured = self._measure_assignment_change(assignments, new_assignments)
            assignments = new_assignments

            iteration += 1

        distances = self._calculate_point_centroid_distance(data, centroids)
        print(f'Score: {self._sum_of_squares_score(data, distances, assignments, centroids)}.')
        return centroids, assignments

    def _randomize_centroids(self, data):
        indices = np.random.randint(0, len(data), size=(self._n_clusters))
        return data[indices, :]

    def _calculate_point_centroid_distance(self, data, centroids):
        distances = np.zeros(shape=(data.shape[0], self._n_clusters))
        for point_ind, point in enumerate(data):
            for centroid_ind, centroid in enumerate(centroids):
                distances[point_ind][centroid_ind] = self._calculate_points_distance(point, centroid)

        return distances
    
    def _calculate_points_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2)) + 0.00001

    def _calculate_mean_centroids(self, data, assignments):
        new_centroids = np.matmul(np.transpose(assignments), data)
        devider = np.sum(assignments, axis=0)

        return new_centroids / devider[:, None]

    def _get_cluster_assignments(self, data, centroids):
        point_centroid_distance = self._calculate_point_centroid_distance(data, centroids)

        assignments = np.zeros(shape=(data.shape[0], self._n_clusters))
        for point_ind, _ in enumerate(data):
            centroids_distance_sum = np.sum(1 / (point_centroid_distance[point_ind, :] ** (2 / (self.FUZZINESS_COEFFICIENT - 1))))
            for centroid_ind, _ in enumerate(centroids):
                powered_dist = point_centroid_distance[point_ind, centroid_ind] ** (2 / (self.FUZZINESS_COEFFICIENT - 1))
                assignments[point_ind, centroid_ind] = 1 / (powered_dist * centroids_distance_sum)

        return assignments

    def _measure_assignment_change(self, old_assignment, new_assignment):
        return np.any(np.abs(old_assignment - new_assignment) > self.EPSILON_THRESHOLD)

    def _normalize_data(self, data):
        data = np.array(data).astype(np.float)
        for i in range(data.shape[1]):
            column_max = np.max(data[:, i])
            data[:, i] *= (self.NORMALIZED_MAX  / column_max)
        
        return data + 0.00001
    
    def _sum_of_squares_score(self, data, distances, assignments, centroids):
        score = 0
        for point_ind, point in enumerate(data):
            for centroid_ind, centroid in enumerate(centroids):
                score += assignments[point_ind, centroid_ind] * (distances[point_ind, centroid_ind] ** 2)
        
        return score
