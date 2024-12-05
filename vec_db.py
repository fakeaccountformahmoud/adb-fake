from typing import Dict, List, Annotated
import numpy as np
import os
import struct
import gc
#from IvfTrain import IvfTrain

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path =  ["/content/adb-fake/saved_clusters.dat", "/content/adb-fake/saved_centroids.dat", "/content/adb-fake/saved_indexes.dat"], new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.cluster_path = index_file_path[0]
        self.centroid_path = index_file_path[1]
        self.index_path = index_file_path[2]
        #print(self.index_path)
        # self._build_index()
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"
    

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def get_all_rows_values(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors[:,1:])

    def get_multiple_rows(self, ranged_clusters_ids):
        ranged_clusters = []
        with open(self.db_path, 'rb') as file:
            for id in ranged_clusters_ids:
                offset = id[1] * DIMENSION * ELEMENT_SIZE
                file.seek(offset)
                packed_data = file.read(DIMENSION * ELEMENT_SIZE)
                unpacked_data = struct.unpack(f'{DIMENSION}f', packed_data)
                del packed_data
                ranged_clusters.append([unpacked_data, id[1]])
                
            file.close()
            del file
            
        
        return ranged_clusters
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        scores = []
        
        centroids = []
        file = open("/content/adb-fake/saved_centroids.dat", 'rb')
        try:
            row_size = ELEMENT_SIZE * (DIMENSION + 1)
            data = file.read()
            #cluster_size = os.path.getsize(self.cluster_path) // ((DIMENSION+1) * ELEMENT_SIZE)
            #print(cluster_size)
            length = len(data)
            for offset in range(0, length, row_size):
                packed_data = data[offset:offset + row_size]
                if len(packed_data) < row_size:
                    break
                #print(len(packed_data), row_size)
                unpacked_data = struct.unpack(f'i{DIMENSION}f', packed_data)
                del packed_data
                
                centroids.append(unpacked_data)
            #print(cnt)
            
        finally:
            # Ensure the file is properly closed
            file.close()
            del file
            
        #print("lol")
        centroids = np.array(centroids)
        # Calculate the dot product between centroids and the query vector
        sorted_indices = np.argsort((np.dot(centroids[:, 1:], query.T).T / (np.linalg.norm(centroids[:, 1:], axis=1) * np.linalg.norm(query))).squeeze())[::-1]

        # Convert to a Python list
        best_centroids = sorted_indices.tolist()

        del sorted_indices
        
        scores = best_centroids[:4]
        del best_centroids
        # print(scores)
        top_k_results = []
        for score in scores:
            first_index, second_index = None, None
            file = open("/content/adb-fake/saved_indexes.dat", 'rb')
            # print(os.path.getsize(self.index_path) // (DIMENSION * ELEMENT_SIZE))
            try:
                # Calculate the byte position based on the score
                position = 3 * score * ELEMENT_SIZE
                # Seek to the calculated position in the file
                file.seek(int(position))
                # Read the data for the indices
                # data = file.read(3 )
                # print(len(data))
                packed_data = file.read(3 * ELEMENT_SIZE)
                # print(len(packed_data),3*ELEMENT_SIZE)
                # Ensure the data read matches the expected size
                # if len(packed_data) < 3 * ELEMENT_SIZE:
                #     print("skill issues")
                #     continue
                # Unpack the data into three integers
                # print(len())
                # print(cnt)
                unpacked_data = struct.unpack('iii', packed_data)
                del packed_data
                first_index, second_index = unpacked_data[1], unpacked_data[2]
                
            finally:
                # Ensure the file is properly closed
                file.close()
                del file
                
            ranged_clusters_ids = []
            with open("/content/adb-fake/saved_clusters.dat", 'rb') as file:
                file.seek(first_index)
                while file.tell() < second_index:
                    packed_data = file.read(2 * ELEMENT_SIZE)
                    if packed_data == b'':
                        break
                    data = struct.unpack('ii', packed_data)
                    del packed_data
                    # print(data)
                    ranged_clusters_ids.append(data)
                    
                file.close()
                del file
                
            
            ranged_clusters = self.get_multiple_rows(ranged_clusters_ids)              #take care
            
            #print(ranged_clusters)
            #ranged_clusters = np.array(ranged_clusters)
            # Compute cosine similarities between the embeddings and the query
            # cosine_similarities = (ranged_clusters[1], ranged_clusters[0].dot(query.T).T / (np.linalg.norm(ranged_clusters[0], axis=1) * np.linalg.norm(query)))
            # print(cosine_similarities)
            # # Pair each similarity score with the corresponding vector ID
            # cluster_best_vectors = [(cosine_similarities[0, i], ranged_clusters_including_id[i][0]) for i in range(len(ranged_clusters_including_id))]
            # cosine_similarities = [data[1]] + cosine_similarities
            cosine_similarities = []
            for row in ranged_clusters:
                cosine_similarity = self._cal_score(query, row[0])
                cosine_similarities.append((cosine_similarity, row[1]))
            

            # print(cosine_similarities)
            # Get the top-k vectors with the highest similarity scores
            # cluster_best_vectors = sorted(cosine_similarities, key=lambda x: x[0], reverse=True)[:50]
            #print(np.array(cluster_best_vectors).shape)
            # Concatenate the top results from all regions
            top_k_results.extend(cosine_similarities)
            del cosine_similarities
            
        #print(np.array(top_k_results).shape)
        print(len(top_k_results))
        scores = sorted(top_k_results, key=lambda x: x[0], reverse=True)[:top_k]
        gc.collect()
        return [s[1] for s in scores]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2).T
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        # Placeholder for index building logic
        lol2 = IvfTrain()

# lol = VecDB(db_size = 20000000, new_db=True)
# lol2 = IvfTrain()
# lol2.build_index()
# rng = np.random.default_rng(DB_SEED_NUMBER)
# vectors = rng.random((1, DIMENSION), dtype=np.float32)
# print(lol.retrieve(vectors))
