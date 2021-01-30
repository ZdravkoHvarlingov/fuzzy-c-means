import csv
from src.fuzzy_c_means import FuzzyCMeans
import numpy as np


def get_columns_of_interest_and_clusters(columns_list):
    print('Available clusterization columns:')
    for ind, column in enumerate(columns_list):
        print(f'{column} - {ind}')
    columns = input("Please specify the ones you want to use. Separate them with a whitespace: ")
    columns = [int(str_num) for str_num in columns.split(' ')]
    print(f'Selected columns: {columns}')

    clusters = int(input('Please, specify the number of clusters: '))

    return columns, clusters


def initial_test():
    with open('data/customer_data.csv') as data_file:
        customer_reader = csv.reader(data_file, delimiter=',')
        header_row = next(customer_reader)
        columns, clusters = get_columns_of_interest_and_clusters(header_row)

        all_data = [row for row in customer_reader]
        extracted_columns_data = [[row[col] for col in columns] for row in all_data]
    
    fuzzy_c_means = FuzzyCMeans(clusters)
    print('\n\n')
    print('##################################################################')
    print('Fuzzy C Means starting...')
    print('##################################################################')
    print('\n\n')
    centroids, assignments = fuzzy_c_means.fit(extracted_columns_data)
    print(assignments[0: 10])


if __name__ == '__main__':
    initial_test()
