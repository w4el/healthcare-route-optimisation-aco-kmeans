# This script performs clustering of patients around GP surgeries using KMeans algorithm.
# It ensures that each GP surgery is included in its cluster and provides visualization of the clustering results.
# Functions:
# 1. cluster_patients: Assigns patients to clusters based on proximity to GP surgeries.
# 2. plot_clusters: Generates a visual representation of the clusters and highlights GP locations.

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# File paths for input data and output results
PATIENT_LOCATIONS_CSV = "./data/processed/patient_locations.csv"
GP_SURGERIES_CSV = "./data/processed/greenwich_gp_surgeries_processed.csv"
CLUSTERED_PATIENTS_CSV = "./data/processed/clustered_patient_locations.csv"

def cluster_patients():
    """
    Assign patients to GP surgeries using KMeans clustering.
    Ensure the GP surgery is always part of its respective cluster.

    Returns:
        pd.DataFrame: Combined dataframe containing GPs and patients with cluster assignments.
    """
    # Load data for GP surgeries and patient locations
    gp_df = pd.read_csv(GP_SURGERIES_CSV)  # GP surgeries with names and coordinates
    patient_df = pd.read_csv(PATIENT_LOCATIONS_CSV)  # Patient locations with coordinates

    # Use GP coordinates as initial cluster centers
    cluster_centers = gp_df[['Latitude', 'Longitude']].to_numpy()

    # Perform KMeans clustering with GP surgeries as predefined cluster centers
    kmeans = KMeans(n_clusters=len(cluster_centers), init=cluster_centers, n_init=1, max_iter=300)
    patient_coords = patient_df[['Latitude', 'Longitude']].to_numpy()
    patient_df['Cluster'] = kmeans.fit_predict(patient_coords)  # Assign each patient to a cluster

    # Assign GP surgeries to their respective clusters
    gp_df['Cluster'] = range(len(gp_df))  # Ensure each GP is assigned a unique cluster
    gp_df['Is_GP'] = True  # Mark GP entries
    gp_df['Assigned_GP'] = gp_df['Name']  # Assign GP names to their clusters

    # Add a unique identifier for each GP surgery
    gp_df['Patient_ID'] = gp_df['Name'] + "_GP"

    # For patients, mark them as not GPs and assign the GP name corresponding to their cluster
    patient_df['Is_GP'] = False
    patient_df['Assigned_GP'] = patient_df['Cluster'].map(gp_df.set_index('Cluster')['Name'])

    # Add a placeholder `Name` column for patients to align with GPs
    patient_df['Name'] = "Patient"

    # Combine GP surgeries and patients into a single dataframe
    combined_df = pd.concat([gp_df, patient_df], ignore_index=True)

    # Reorder columns for better readability
    combined_df = combined_df[['Patient_ID', 'Name', 'Latitude', 'Longitude', 'Assigned_GP', 'Is_GP', 'Cluster']]

    # Save the combined dataframe to a CSV file
    combined_df.to_csv(CLUSTERED_PATIENTS_CSV, index=False)
    print(f"Clustered data saved to {CLUSTERED_PATIENTS_CSV}")

    return combined_df  # Return the combined dataframe for further processing

def plot_clusters(combined_df):
    """
    Plot clusters of patients and GP surgeries.

    Args:
        combined_df (pd.DataFrame): Dataframe containing patients and GP surgeries with cluster assignments.
    """
    # Initialize a plot
    plt.figure(figsize=(10, 8))

    # Plot each cluster with a unique color
    for cluster in combined_df['Cluster'].unique():
        cluster_data = combined_df[combined_df['Cluster'] == cluster]
        plt.scatter(cluster_data['Longitude'], cluster_data['Latitude'], label=f'Cluster {cluster}', alpha=0.6)
    
    # Highlight GP surgeries in the plot
    gp_data = combined_df[combined_df['Is_GP'] == True]
    plt.scatter(gp_data['Longitude'], gp_data['Latitude'], color='red', s=100, marker='*', label='GP Surgeries')

    # Add labels, title, and legend
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('KMeans Clustering of Patients and GP Surgeries')
    plt.legend()

    # Save the plot to a file
    plt.savefig('clusters_visualization.png')
    plt.show()  # Display the plot

if __name__ == "__main__":
    # Run clustering and visualization
    combined_df = cluster_patients()  # Perform clustering and get the combined dataframe
    plot_clusters(combined_df)  # Visualize the clustering results
