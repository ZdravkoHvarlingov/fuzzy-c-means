import csv
import yaml
from src.fuzzy_c_means import FuzzyCMeans
import numpy as np


def extract_csv_data():
    with open('data/customer_data.csv') as data_file:
        customer_reader = csv.reader(data_file, delimiter=',')
        next(customer_reader)
        
        return [row for row in customer_reader]


def extract_config():
    with open('config/config.yml', 'r') as yml_file:
        return yaml.safe_load(yml_file)

def initial_test():

    data = extract_csv_data()
    config = extract_config()

    fuzzy_c_means = FuzzyCMeans(config.get('n_clusters'))
    print('\n\n')
    print('##################################################################')
    print('Fuzzy C Means starting...')
    print('##################################################################')
    print('\n\n')
    centroids, assignments = fuzzy_c_means.fit(data, config.get('columns'))
    # print(assignments[0: 10])


if __name__ == '__main__':
    initial_test()
