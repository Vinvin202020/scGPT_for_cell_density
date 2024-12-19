# Import necessary libraries for data manipulation, numerical calculations, sparse matrices, and spatial operations
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix  # Efficient sparse matrix for large datasets
import time  # Used to measure execution time
from scipy.spatial.distance import pdist, squareform  # For distance calculations between points

# Load the Xenium data from a CSV file located at '/data/cells.csv.gz'
filename = '/data/cells.csv.gz'
cells = pd.read_csv(filename)  # Read the CSV into a pandas DataFrame

# Extract the x and y coordinates of the cells from the DataFrame
column_x = cells.iloc[:, 1].values  # The x-coordinates of the cells
column_y = cells.iloc[:, 2].values  # The y-coordinates of the cells

# Combine x and y coordinates into a single matrix of positions for each cell
matrix_pos = np.column_stack((column_x, column_y))
print(matrix_pos)  # Print the position matrix for verification

# Function to compute the adjacency matrix for the cells based on their proximity
def compute_adjacency_matrix(positions, delta):
    n = positions.shape[0]  # Number of cells (positions)
    adjacency_matrix = lil_matrix((n, n), dtype=int)  # Initialize a sparse matrix to store adjacency info

    start_time = time.time()  # Start the timer to measure computation time

    # Loop over each pair of cells to compute the Euclidean distance
    for i in range(n):
        elapsed_time = time.time() - start_time  # Calculate elapsed time for progress tracking
        print(f"\rIteration {i} | Elapsed Time: {elapsed_time:.2f} seconds", end='', flush=True)  # Print progress

        for j in range(i + 1, n):  # Only check the upper triangle of the matrix to avoid redundant comparisons
            distance = np.linalg.norm(positions[i] - positions[j])  # Compute the Euclidean distance between cell i and j
            if distance <= delta:  # If distance is smaller than or equal to the threshold (delta)
                adjacency_matrix[i, j] = 1  # Set adjacency (i, j) to 1 (cells are connected)
                adjacency_matrix[j, i] = 1  # Symmetric adjacency (j, i)

    return adjacency_matrix  # Return the computed adjacency matrix (sparse)

# Function to calculate cell density based on the adjacency matrix
def calculate_density(positions, delta):
    adj_matrix = compute_adjacency_matrix(positions, delta)  # Compute the adjacency matrix
    return np.array(adj_matrix.sum(axis=1)).flatten()  # Sum the rows of the adjacency matrix to get cell density

# Initialize an empty array to store the density values
density_array = np.zeros(matrix_pos.shape[0])

delta_r = 20  # Define the threshold distance for adjacency (change as needed)

# Compute the density for all cells using the specified delta_r value
density_array = calculate_density(matrix_pos, delta_r)

# Create a DataFrame to store the resulting density values
df = pd.DataFrame(density_array, columns=['Value'])

# Save the DataFrame to a CSV file
df.to_csv('cell_density_20.csv', index=True)

print("Result vector saved to cell_density_20.csv")  # Confirmation message
