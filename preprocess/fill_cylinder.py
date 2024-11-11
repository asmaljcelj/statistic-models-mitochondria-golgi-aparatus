import os
import pandas as pd
from sklearn.decomposition import PCA

instances_folder = '../ga_instances'

pca = PCA(n_components=2)
for file in os.listdir(instances_folder):
    print('processing file', file)
    dataset = pd.read_csv(instances_folder + '/' + file)
    pca.fit(dataset)
    eig_vec = pca.components_
    print()

