from typing import Dict, List, Annotated
import numpy as np
import os
import struct
from IvfTrain import IvfTrain
#from sortedcontainers import SortedList
import heapq
#from itertools import chain

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index_path", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.general_path = index_file_path
        self.size = 2
        self.db_size = db_size
        # print(index_file_path[0])
        # print(index_file_path[1])
        # print(index_file_path[2])
        #print(self.index_path)
        #self._build_index()
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
        # print(ranged_clusters_ids)
        with open(self.db_path, 'rb') as file:
            for id in ranged_clusters_ids:
                #print(id)
                offset = np.int64(id[1]) * DIMENSION * ELEMENT_SIZE
                # print(id[1])
                # print(offset)
                file.seek(offset)
                packed_data = file.read(DIMENSION * ELEMENT_SIZE)
                unpacked_data = struct.unpack(f'{DIMENSION}f', packed_data)
                del packed_data
                ranged_clusters.append([unpacked_data, id[1]])
            file.close()
            del file
        return ranged_clusters
    
    def divide_into_batches(self, arr, batch_size):
        length = len(arr)
        return [arr[i:i + batch_size] for i in range(0, length, batch_size)]
    
    def ceil(self, x):
        integer_part = x - (x % 1)  # Extract integer part
        return integer_part + (1 if x % 1 > 0 else 0)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        scores = []
        
        centroids = []
        file = open(f"{self.general_path}/saved_centroids.dat", 'rb')
        try:
            row_size = ELEMENT_SIZE * (DIMENSION + 1)
            data = file.read()
            length = len(data)
            for offset in range(0, length, row_size):
                if len(data[offset:offset + row_size]) < row_size:
                    break
                unpacked_data = struct.unpack(f'i{DIMENSION}f', data[offset:offset + row_size])
                centroids.append(unpacked_data)
        finally:
            file.close()
            del file
        centroids = np.array(centroids)
        sorted_indices = np.argsort((np.dot(centroids[:, 1:], query.T).T / (np.linalg.norm(centroids[:, 1:], axis=1) * np.linalg.norm(query))).squeeze())[::-1]
        best_centroids = sorted_indices.tolist()

        del sorted_indices
        
        if self._get_num_records() == 20000000:
            #print(top_k)
            scores = best_centroids[:10]
        elif self._get_num_records() == 15000000:
            scores = best_centroids[:40]
        elif self._get_num_records() == 10000000:
            scores = best_centroids[:50]
        else:
            scores = best_centroids[:60]
        del best_centroids
        top_k_results = []
        for score in scores:
            first_index, second_index = None, None
            file = open(f"{self.general_path}/saved_indexes.dat", 'rb')
            try:
                position = 3 * score * ELEMENT_SIZE
                file.seek(int(position))
                packed_data = file.read(3 * ELEMENT_SIZE)

                unpacked_data = struct.unpack('iii', packed_data)
                del packed_data
                first_index, second_index = unpacked_data[1], unpacked_data[2]
            finally:

                file.close()
                del file
            ranged_clusters_ids = []
            with open(f"{self.general_path}/saved_clusters.dat", 'rb') as file:
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

            # batch_size = int(0.1*len(ranged_clusters_ids)) + 1
            # length = len(ranged_clusters_ids)
            # ranged_clusters_ids = np.array(ranged_clusters_ids)
            # top_k_results = []
            # for i in range(0, length, batch_size):
            #     ranged_clusters = self.get_multiple_rows(ranged_clusters_ids[i:min(i + batch_size, length)])
            #     for row in ranged_clusters:
            #         cosine_similarity = self._cal_score(query, row[0])
            #         #print(cosine_similarity, "/n")
            #         top_k_results.append((cosine_similarity, row[1]))
            #     #print(ranged_clusters)
            #     del ranged_clusters
            ranged_clusters = self.get_multiple_rows(ranged_clusters_ids)
            best_vectors = []
            for row in ranged_clusters:
                cosine_similarity = self._cal_score(query, row[0])
                #print(cosine_similarity, "/n")
                if len(best_vectors) < top_k:
                    heapq.heappush(best_vectors, (cosine_similarity, row[1]))  # Push new element
                else:
                    heapq.heappushpop(best_vectors, (cosine_similarity, row[1]))
            length = len(ranged_clusters_ids)
            top_k_results.extend(sorted(best_vectors, key=lambda x: x[0], reverse=True))
            del length
            del best_vectors
        # print(len(top_k_results))
        scores = sorted(top_k_results, key=lambda x: x[0], reverse=True)[:top_k]
        # print([s[1] for s in scores])
        return [s[1] for s in scores]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2).T
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        del dot_product
        del norm_vec1
        del norm_vec2
        return cosine_similarity

    def _build_index(self):
        # Placeholder for index building logic
        lol2 = IvfTrain()
