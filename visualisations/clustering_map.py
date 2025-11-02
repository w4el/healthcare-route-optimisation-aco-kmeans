import folium
import pandas as pd
from PIL import Image
import io

# Paths for input and output
ACO_RESULTS_CSV = "./data/processed/aco_results_with_coordinates.csv"
CLUSTERED_PATIENTS_CSV = "./data/processed/clustered_patient_locations.csv"
FOLIUM_MAP_OUTPUT = "./visualisations/all_clusters_aco_map.html"
THUMBNAIL_IMAGE = "./visualisations/thumbnail.png"
REPRESENTATIVE_IMAGE = "./visualisations/representative_image.png"

def visualize_all_clusters_on_one_map():
    """
    Generate a single Folium map showing all clusters, GPs, and ACO paths.
    """
    # Load ACO results and clustered patient data
    aco_results = pd.read_csv(ACO_RESULTS_CSV)
    patient_data = pd.read_csv(CLUSTERED_PATIENTS_CSV)

    # Initialize a Folium map centered on the average coordinates of all GP surgeries
    gp_coords = patient_data[patient_data['Is_GP'] == True][['Latitude', 'Longitude']].mean().tolist()
    folium_map = folium.Map(location=gp_coords, zoom_start=13)

    # Process each GP's ACO path
    for _, row in aco_results.iterrows():
        gp_name = row["GP_Name"]

        # Parse ACO path coordinates
        best_path_coordinates = [
            tuple(map(float, coord.split(','))) for coord in row["Best_Path_Coordinates"].split(';')
        ]

        # Extract GP location from cluster data
        gp_data = patient_data[(patient_data['Is_GP'] == True) & (patient_data['Assigned_GP'] == gp_name)]
        if gp_data.empty:
            print(f"Warning: No GP found for {gp_name}. Skipping.")
            continue
        gp_location = [gp_data.iloc[0]['Latitude'], gp_data.iloc[0]['Longitude']]

        # Add GP marker using accurate GP location
        folium.Marker(
            location=gp_location,
            popup=f"GP Surgery: {gp_name}",
            icon=folium.Icon(color="red", icon="plus-sign"),
        ).add_to(folium_map)

        # Add ACO path (fully connected, including return to the starting point)
        full_path_coordinates = best_path_coordinates + [best_path_coordinates[0]]  # Close the loop
        folium.PolyLine(
            locations=full_path_coordinates,
            color="green",
            weight=3,
            opacity=0.7,
        ).add_to(folium_map)

        # Add patient markers for this GP's cluster
        cluster_patients = patient_data[patient_data['Assigned_GP'] == gp_name]
        for _, patient in cluster_patients.iterrows():
            folium.CircleMarker(
                location=[patient['Latitude'], patient['Longitude']],
                radius=5,
                color="blue" if not patient['Is_GP'] else "red",
                fill=True,
                fill_color="blue" if not patient['Is_GP'] else "red",
                fill_opacity=0.7,
                popup=(f"Patient ID: {patient['Patient_ID']}<br>"
                       f"Assigned GP: {patient['Assigned_GP']}<br>"
                       f"Cluster: {patient['Cluster']}"),
            ).add_to(folium_map)

    # Save the map
    folium_map.save(FOLIUM_MAP_OUTPUT)
    print(f"ACO path map saved to {FOLIUM_MAP_OUTPUT}")

    # Capture the full map as an image (for 1000×1000 px)
    img_data = folium_map._to_png(5)
    img = Image.open(io.BytesIO(img_data))
    img = img.resize((1000, 1000), Image.LANCZOS)  # Use LANCZOS for high-quality resizing
    img.save(REPRESENTATIVE_IMAGE)
    print(f"Representative image saved to {REPRESENTATIVE_IMAGE}")

    # Create the thumbnail (250×250 px)
    thumbnail = img.resize((250, 250), Image.LANCZOS)  # Use LANCZOS for resizing
    thumbnail.save(THUMBNAIL_IMAGE)
    print(f"Thumbnail image saved to {THUMBNAIL_IMAGE}")

if __name__ == "__main__":
    visualize_all_clusters_on_one_map()
