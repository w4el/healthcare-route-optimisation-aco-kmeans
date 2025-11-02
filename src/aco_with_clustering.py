# This file contains the implementation of Ant Colony Optimization (ACO) integrated with clustering.
# It performs optimization for travel routes while grouping patients to their assigned GP surgeries.
# Functions:
# 1. calculate_distance_matrix: Computes pairwise distances between points in a cluster.
# 2. run_aco_with_visualization: Runs ACO for a single cluster and visualizes the optimization process.
# 3. run_aco_for_clusters: Executes the optimization for all clusters, saving results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from geopy.distance import geodesic
from aco import ACO, Graph

# File paths for input data and results
CLUSTERED_PATIENTS_CSV = "./data/processed/clustered_patient_locations.csv"
ACO_RESULTS_CSV = "./data/processed/aco_results_with_coordinates.csv"

def calculate_distance_matrix(cluster):
    """
    Compute the distance matrix for a cluster of points (patients and GP).
    
    Args:
        cluster (pd.DataFrame): Dataframe containing 'Latitude' and 'Longitude' of points.

    Returns:
        np.ndarray: Symmetric distance matrix where entry (i, j) represents the distance 
                    between points i and j in kilometers.
    """
    coords = cluster[['Latitude', 'Longitude']].to_numpy()
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = geodesic(coords[i], coords[j]).kilometers
    return distance_matrix

def run_aco_with_visualization(cluster, gp_name, aco_params):
    """
    Run ACO for a single cluster with real-time visualization and convergence tracking.

    Args:
        cluster (pd.DataFrame): Dataframe containing points in the cluster.
        gp_name (str): Name of the GP surgery.
        aco_params (dict): Parameters for the ACO algorithm.

    Returns:
        tuple: Best solution (path), best cost (distance), and best path coordinates.
    """
    # Calculate distance matrix for the cluster
    distance_matrix = calculate_distance_matrix(cluster)
    graph = Graph(distance_matrix, len(cluster))
    aco = ACO(**aco_params)

    # Extract coordinates for visualization
    coords = cluster[['Latitude', 'Longitude']].to_numpy()

    # Initialize matplotlib for visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"ACO for GP: {gp_name}")
    ax.scatter(coords[:, 1], coords[:, 0], c='blue', label='Patients')
    ax.scatter(coords[0, 1], coords[0, 0], c='red', s=100, label='GP Surgery', marker='*')
    ax.legend()

    # Track best solution and costs for convergence graph
    best_solution = None
    best_cost = float('inf')
    lines = []
    best_costs = []  # Store costs across iterations

    def update(frame):
        nonlocal best_solution, best_cost

        # Clear previous lines
        while lines:
            line = lines.pop()
            line.remove()

        # Run ACO for one generation
        solution, cost = aco.solve(graph)

        # Update best solution if cost is improved
        if cost < best_cost:
            best_cost = cost
            best_solution = solution

        best_costs.append(best_cost)

        # Plot current solution
        for i in range(len(solution) - 1):
            start = coords[solution[i]]
            end = coords[solution[i + 1]]
            line, = ax.plot([start[1], end[1]], [start[0], end[0]], c='green', alpha=0.5)
            lines.append(line)

        # Close the loop
        start = coords[solution[-1]]
        end = coords[solution[0]]
        line, = ax.plot([start[1], end[1]], [start[0], end[0]], c='green', alpha=0.5, linestyle='--')
        lines.append(line)

        # Update title with generation and best cost
        ax.set_title(f"ACO for GP: {gp_name} - Iteration: {frame + 1}, Best Cost: {best_cost:.2f} km")

    # Animate the ACO process
    ani = FuncAnimation(fig, update, frames=aco_params["generations"], repeat=False)
    plt.show()

    # Plot convergence graph
    plt.figure()
    plt.plot(range(len(best_costs)), best_costs, label='Best Cost per Iteration', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (km)')
    plt.title(f'Convergence of ACO for GP: {gp_name}')
    plt.legend()
    plt.savefig(f'aco_convergence_{gp_name}.png')
    plt.show()

    # Save final best solution coordinates
    best_path_coordinates = [
        {
            "Latitude": cluster.iloc[node]["Latitude"],
            "Longitude": cluster.iloc[node]["Longitude"],
            "Name": cluster.iloc[node].get("Name", "Patient")
        }
        for node in best_solution
    ]
    return best_solution, best_cost, best_path_coordinates


def run_aco_for_clusters():
    """
    Run ACO for each GP cluster, visualize the process, and save the results.
    """
    # Load the clustered patient data
    data = pd.read_csv(CLUSTERED_PATIENTS_CSV)
    results = []

    # ACO algorithm parameters
    aco_params = {
        "ant_count": 20,         # Number of ants (recommended >= cluster size)
        "generations": 50,       # Number of iterations
        "alpha": 1.0,            # Importance of pheromone
        "beta": 1.5,             # Importance of heuristic information
        "rho": 0.3,              # Pheromone evaporation rate
        "q": 100,                # Pheromone deposit factor
        "strategy": 0            # Ant-cycle update strategy
    }

    # Process each cluster for the assigned GP surgery
    for gp_name in data['Assigned_GP'].unique():
        cluster = data[data['Assigned_GP'] == gp_name].reset_index(drop=True)

        # Ensure GP is the first point in the cluster
        gp = cluster[cluster['Is_GP'] == True]
        cluster = pd.concat([gp, cluster[cluster['Is_GP'] == False]]).reset_index(drop=True)

        # Run ACO for the cluster with visualization
        best_solution, best_cost, best_path_coordinates = run_aco_with_visualization(cluster, gp_name, aco_params)

        print(f"Best route for {gp_name}: {best_solution}, Length: {best_cost:.2f} km")

        # Save results for this GP
        results.append({
            "GP_Name": gp_name,
            "Best_Route": best_solution,
            "Best_Path_Coordinates": best_path_coordinates,
            "Total_Distance": best_cost
        })

    # Save all results to a CSV file
    results_df = pd.DataFrame(results)
    results_df["Best_Path_Coordinates"] = results_df["Best_Path_Coordinates"].apply(
        lambda x: ";".join([f"{coord['Latitude']},{coord['Longitude']}" for coord in x])
    )
    results_df.to_csv(ACO_RESULTS_CSV, index=False)
    print(f"ACO results saved to {ACO_RESULTS_CSV}")


if __name__ == "__main__":
    run_aco_for_clusters()
