import numpy as np
import os
from vec_db import VecDB
import time
from dataclasses import dataclass
from typing import List
from memory_profiler import memory_usage
import gc

@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

def run_queries(db : VecDB, queries, top_k, actual_ids, num_runs):
    """
    Run queries on the database and record results for each query.

    Parameters:
    - db: Database instance to run queries on.
    - queries: List of query vectors.
    - top_k: Number of top results to retrieve.
    - actual_ids: List of actual results to evaluate accuracy.
    - num_runs: Number of query executions to perform for testing.

    Returns:
    - List of Result
    """
    global results_a
    results_a = []
    for i in range(num_runs):
        tic = time.time()
        print("starting searching...")
        db_ids = db.retrieve(queries[i], top_k)
        toc = time.time()
        run_time = toc - tic
        print(f"done searching in: {run_time}")
        results_a.append(Result(run_time, top_k, db_ids, actual_ids[i]))
    return results_a

def memory_usage_run_queries(args):
    """
    Run queries and measure memory usage during the execution.

    Parameters:
    - args: Arguments to be passed to the run_queries function.

    Returns:
    - results: The results of the run_queries.
    - memory_diff: The difference in memory usage before and after running the queries.
    """
    global results_a
    mem_before = max(memory_usage())
    mem = memory_usage(proc=(run_queries, args, {}), interval = 1e-3)
    return results_a, max(mem) - mem_before

def evaluate_result(results_a: List[Result]):
    """
    Evaluate the results based on accuracy and runtime.
    Scores are negative. So getting 0 is the best score.

    Parameters:
    - results: A list of Result objects

    Returns:
    - avg_score: The average score across all queries.
    - avg_runtime: The average runtime for all queries.
    """
    scores = []
    run_time = []
    for res in results_a:
        run_time.append(res.run_time)
        # case for retireving number not equal to top_k, socre will be the lowest
        if len(set(res.db_ids)) != res.top_k or len(res.db_ids) != res.top_k:
            scores.append( -1 * len(res.actual_ids) * res.top_k)
            continue
        score = 0
        for id in res.db_ids:
            try:
                ind = res.actual_ids.index(id)
                if ind > res.top_k * 3:
                    score -= ind
            except:
                score -= len(res.actual_ids)
        scores.append(score)

    return sum(scores) / len(scores), sum(run_time) / len(run_time)

def get_actual_ids_first_k(actual_sorted_ids, k):
    """
    Retrieve the IDs from the sorted list of actual IDs.
    actual IDs has the top_k for the 20 M database but for other databases we have to remove the numbers higher than the max size of the DB.

    Parameters:
    - actual_sorted_ids: A list of lists containing the sorted actual IDs for each query.
    - k: The DB size.

    Returns:
    - List of lists containing the actual IDs for each query for this DB.
    """
    return [[id for id in actual_sorted_ids_one_q if id < k] for actual_sorted_ids_one_q in actual_sorted_ids]

if __name__ == "__main__":
    db = VecDB(db_size = 20000000, new_db = False)

    needed_top_k = 10000
    rng = np.random.default_rng(10)
    query1 = rng.random((1, 70), dtype=np.float32)
    query2 = rng.random((1, 70), dtype=np.float32)
    query3 = rng.random((1, 70), dtype=np.float32)
    query_dummy = rng.random((1, 70), dtype=np.float32)

    print("fetching all data...")
    vectors = db.get_all_rows()
    print("done fetching")

    actual_sorted_ids_20m_q1 = np.argsort(vectors.dot(query1.T).T / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query1)), axis= 1).squeeze().tolist()[::-1][:needed_top_k]
    gc.collect()
    actual_sorted_ids_20m_q2 = np.argsort(vectors.dot(query2.T).T / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query2)), axis= 1).squeeze().tolist()[::-1][:needed_top_k]
    gc.collect()
    actual_sorted_ids_20m_q3 = np.argsort(vectors.dot(query3.T).T / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query3)), axis= 1).squeeze().tolist()[::-1][:needed_top_k]
    gc.collect()

    queries = [query1, query2, query3]
    actual_sorted_ids_20m = [actual_sorted_ids_20m_q1, actual_sorted_ids_20m_q2, actual_sorted_ids_20m_q3]
    actual_ids = get_actual_ids_first_k(actual_sorted_ids_20m, 20*(10**6))
    # Make a dummy run query to make everything fresh and loaded (wrap up)
    res = run_queries(db, query_dummy, 5, actual_ids, 1)
    # actual runs to evaluate
    res, mem = memory_usage_run_queries((db, queries, 5, actual_ids, 3))
    eval = evaluate_result(res)
    to_print = f"score\t{eval[0]}\ttime\t{eval[1]:.2f}\tRAM\t{mem:.2f} MB"
    print(to_print)
    del db
    del actual_ids
    del res
    del mem
    del eval
    gc.collect()