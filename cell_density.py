import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
import time
from scipy.spatial.distance import pdist, squareform

filename = '/home/auberger/xenium_data/cells.csv.gz'  #load xenium data
cells = pd.read_csv('/home/auberger/xenium_data/cells.csv.gz')

column_x = cells.iloc[:, 1].values
column_y = cells.iloc[:, 2].values

matrix_pos = np.column_stack((column_x, column_y))
print(matrix_pos)


def compute_adjacency_matrix(positions, delta):
    n = positions.shape[0]
    adjacency_matrix = lil_matrix((n, n), dtype=int) #with a sparse matrix to reduce cost
    start_time = time.time()

    for i in range(n):
        elapsed_time = time.time() - start_time
        print(f"\rIteration {i} | Elapsed Time: {elapsed_time:.2f} seconds", end='', flush=True)
        for j in range(i + 1, n):  # Check only the upper triangle:
                distance = np.linalg.norm(positions[i] - positions[j])  # Euclidean distance
                if distance <= delta:
                   adjacency_matrix[i, j] = 1
                   adjacency_matrix[j, i] = 1  # Symmetric


    #####################################################################################
    # Other method much faster but problem with memory allocation
    #distances = np.zeros(n*n, np.float32)
    #distances = pdist(positions)  # Returns a 1D array of distances
    #print("distances pairwise calculated")

    # Convert the condensed distance matrix to a square form
    #distance_matrix = squareform(distances)

    # Create the adjacency matrix: 1 where distance <= delta, 0 otherwise
    #adjacency_matrix = (distance_matrix <= delta).astype(np.uint8)

    # Remove self-connectivity (diagonal elements)
    #np.fill_diagonal(adjacency_matrix, 0)

    return adjacency_matrix

def calculate_density(positions, delta):
    #Compute adjacency matrix
    adj_matrix = compute_adjacency_matrix(positions, delta)
    #Calculate and return cell_density vector
    return np.array(adj_matrix.sum(axis=1)).flatten()


density_array = np.zeros(matrix_pos.shape[0])
delta_r = 20  #to change
density_array = calculate_density(matrix_pos, delta_r)
df = pd.DataFrame(density_array, columns=['Value'])

# Write the DataFrame to a CSV file
df.to_csv('cell_density_20.csv', index=True)

print("Result vector saved to cell_density_20.csv")
