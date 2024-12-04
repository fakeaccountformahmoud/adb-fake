from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.cluster.vq import kmeans2

import random
import time
import os
import numpy as np
import struct
from pathlib import Path
number_of_training_data = int(1e4)

DIMENSION = 70

class IvfTrain:
    def __init__(self, generated_database = "saved_db.dat", clusters = "saved_clusters.dat", centroids = "saved_centroids.dat", indexes = "saved_indexes.dat"):
        self.generated_database = generated_database
        self.clusters = clusters
        self.centroids = centroids
        self.indexes = indexes
        self.dimension = 70
        self.build_index()
        # self.training_data_number()
        
    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * self.dimension * np.dtype(np.float32).itemsize
            mmap_vector = np.memmap(self.generated_database, dtype=np.float32, mode='r', shape=(1, self.dimension), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.generated_database, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def training_data_number(self):
        self.training_data = number_of_training_data
        return self.training_data
    
    def _get_num_records(self) -> int:
        return os.path.getsize(self.generated_database) // (self.dimension * np.dtype(np.float32).itemsize)

    def process_clusters_and_centroids(self, centroids, clusters):
    # Prepare centroid data for insertion
        centroid_records = []
        for idx, vector in enumerate(centroids):
            centroid_records.append({"id": idx, "embed": vector})
        
        # Insert centroids into the database
        #self.bfh_cen.insert_records(centroid_records)
        
        start_offset, end_offset = None, None
        file = open(self.centroids, 'ab')
        try:
            start_offset = file.tell()
            for record in centroid_records:
                record_id, embedding = record["id"], record["embed"]
                binary_record = struct.pack(f'i{self.dimension}f', record_id, *embedding)
                file.write(binary_record)
            end_offset = file.tell()
        finally:
            file.close()
            del file

        for cluster_id, vectors in enumerate(clusters):
            vector_count = len(vectors)
            # print(f"Cluster {cluster_id} contains {vector_count} vectors.")
        print(self._get_num_records())

    def file_output(self, clusters, centroids):
        Path(self.clusters).touch()
        Path(self.centroids).touch()
        Path(self.indexes).touch()
        for idx, vector in enumerate(clusters):
            file = open(self.clusters, 'ab')
            start_offset, end_offset = None, None
            try:
                start_offset = file.tell()
                # print("start", start_offset)
                for row in vector:
                    # print(idx)
                    # print(row)

                    binary_data = struct.pack('ii', idx, row)
                    file.write(binary_data)
                end_offset = file.tell()
                # print("end", end_offset)
            finally:
                file.close()
                del file
            file = open(self.indexes, 'ab')
            try:
                binary_data = struct.pack('iii', idx, start_offset, end_offset)
                file.write(binary_data)
            finally:
                file.close()
                del file
        self.process_clusters_and_centroids(centroids, clusters)
        
        
            

    def build_index(self):
        start_time = time.time()

        #   Here, we should train a sample of the given databse, as
        # training the whole 20 million vectors is very difficult given the constarints

        # print(min(self._get_num_records(), self.training_data)-1)
        # print(len(self.generate_database))
        # training_vector_indexes = random.sample(range(self._get_num_records()), min(self._get_num_records(), self.training_data))

        # training_vector = [self.get_one_row(row) for row in training_vector_indexes]
        # tt = np.array(training_vector)
        # print(tt.shape)
        # for idx in range(len(training_vector)):
        #     # Access the row explicitly using the index
        #     row = training_vector[idx]
        #     # Prepend the index to the row
        #     training_vector[idx] = [idx] + row.tolist()    
        # training_vector = np.array(training_vector)
        # # number if clusters
        # print(training_vector.shape)
        # training_vector_trimmed = [tuple(data) for data in training_vector[:,1:]]
        # # print(np.array(training_vector_trimmed).shape)
        # k = 100
        # chunk_size = self._get_num_records() // k
        # # print(len(training_vector))
        # centroids, labels = kmeans2(training_vector_trimmed, k)
        # centroids = centroids.tolist()

        #kmeans -> O(n^2)
        #(50,000)^2



        # clusters = [[] for i in range(k)]
        # print(self._get_num_records())
        # for chunk_idx in range(k):
        #     start_id = chunk_idx * chunk_size
        #     end_id = start_id + chunk_size
        #     feature_vectors = [self.get_one_row(row_id) for row_id in range(start_id, end_id)]
        #     for idx in range(len(feature_vectors)):
        #         # Access the row explicitly using the index
        #         row = feature_vectors[idx]
        #         # Prepend the index to the row
        #         feature_vectors[idx] = [start_id + idx] + row.tolist()   
        #     # print(np.array(feature_vectors).shape)
        #     #feature_vectors = [tuple(row[1:]) for row in id_feature]
        #     # print(len(feature_vectors[99]))
        #     # print(feature_vectors[0])
        #     # print(feature_vectors[434])
        #     # print(id_feature[0])
        #     # print(feature_vectors[0])
        #     predictions = kmeans2([tuple(row[1:]) for row in feature_vectors], centroids, minit="matrix")[1]
        #     # print(predictions)
        #     # print(predictions.shape)
        #     for idx, cluster_id in enumerate(predictions):
        #         # print(cluster_id)
        #         # print(int(feature_vectors[idx][0]))
        #         clusters[int(cluster_id)].append(int(feature_vectors[idx][0]))       #casted to integer after it gave me an error for two hours straight in the file insertion  :D

        #print("Cluster Labels:\n", clusters[9])
        # print("Centroids:\n", centroids.shape)
        # print(clusters)
        # self.file_output(clusters,centroids)


        # np.random.seed(42)

        # Number of clusters
        n_clusters = 1000
        print("Starting kmeans")
        # Initialize and fit MiniBatchKMeans
        mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000)
        print("Done mini batch")
        hmm = self.get_all_rows()
        print("We are in boys")

        mbk.fit(hmm)
        print("Done fitting")
        end_time_meh = time.time()
        print(f"Time spent: {end_time_meh-start_time:.6f} seconds")
        centroids = mbk.cluster_centers_
        clusters_ids = mbk.labels_
        #print(clusters_ids.shape)
        clusters = [[] for i in range(n_clusters)]
        for idx, cluster_id in enumerate(clusters_ids):
            #print(cluster_id)
            #print(int(feature_vectors[idx][0]))
            clusters[int(cluster_id)].append(int(idx))
            # print(cluster_id)    
        print("training done, now file outputting")

        # print(clusters)

        # Cluster centers
        # print("Cluster Centers:")
        # print(mbk.cluster_centers_)

        # # Predict cluster for new data
        # new_data = np.array([[0.1, 0.2], [0.8, 0.9]])
        # predicted_clusters = mbk.predict(new_data)
        # print("Predicted clusters for new data:", predicted_clusters)

        self.file_output(clusters, centroids)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time spent: {elapsed_time:.6f} seconds")