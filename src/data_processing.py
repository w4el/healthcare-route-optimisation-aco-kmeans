# This script processes GP surgery locations from a GeoJSON file and generates synthetic patient locations
# within a specified radius around each GP surgery. The data is saved in CSV format for further use.

import geopandas as gpd
import pandas as pd
from geopy.distance import geodesic
from geopy.point import Point
import random

# Paths for input and output files
GEOJSON_FILE = "./data/raw/greenwich_gp_surgeries.geojson"  # Input GeoJSON file containing GP surgery locations
GP_SURGERIES_CSV = "./data/processed/greenwich_gp_surgeries_processed.csv"  # Output CSV for processed GP data
PATIENT_LOCATIONS_CSV = "./data/processed/patient_locations.csv"  # Output CSV for generated patient data

def generate_random_points(center_point, radius_km, num_points):
    """
    Generate random points (latitude, longitude) around a given center point within a specified radius.

    Args:
        center_point (Point): The central geographic point (latitude, longitude).
        radius_km (float): The radius (in kilometers) within which to generate random points.
        num_points (int): Number of random points to generate.

    Returns:
        list: A list of tuples representing the generated (latitude, longitude) points.
    """
    points = []
    for _ in range(num_points):
        distance = random.uniform(0, radius_km)  # Random distance from the center within the radius
        angle = random.uniform(0, 360)  # Random angle in degrees
        new_point = geodesic(kilometers=distance).destination(center_point, angle)  # Calculate new point
        points.append((new_point.latitude, new_point.longitude))  # Append the generated coordinates
    return points

def process_geojson(file_path):
    """
    Extract GP surgery details (name, latitude, longitude) from a GeoJSON file.

    Args:
        file_path (str): Path to the GeoJSON file.

    Returns:
        pd.DataFrame: A DataFrame containing the processed GP surgery details.
    """
    # Load the GeoJSON file using GeoPandas
    gp_data = gpd.read_file(file_path)

    # Extract relevant details for GP surgeries
    gp_surgeries = []
    for _, row in gp_data.iterrows():
        if row.geometry.geom_type == 'Point':  # Ensure the geometry is a point
            name = row.get('name', 'Unknown')  # Extract the 'name' field or assign 'Unknown' if missing
            longitude, latitude = row.geometry.x, row.geometry.y  # Extract coordinates
            gp_surgeries.append({'Name': name, 'Latitude': latitude, 'Longitude': longitude})
    
    # Convert the extracted details into a DataFrame
    gp_df = pd.DataFrame(gp_surgeries)
    return gp_df

def generate_patient_data(gp_df, radius_km=0.5, patients_per_gp=20):
    """
    Generate synthetic patient data within a specified radius around each GP surgery.

    Args:
        gp_df (pd.DataFrame): DataFrame containing GP surgery details (name, latitude, longitude).
        radius_km (float): Radius (in kilometers) for generating patient locations.
        patients_per_gp (int): Number of patients to generate per GP surgery.

    Returns:
        pd.DataFrame: A DataFrame containing synthetic patient data.
    """
    patient_data = []  # List to store generated patient data
    patient_counter = 1  # Counter for generating unique patient IDs

    for _, row in gp_df.iterrows():
        gp_location = Point(row['Latitude'], row['Longitude'])  # Create a geopy Point object for the GP location
        gp_name = row['Name'] if row['Name'] else "Unknown"  # Use GP name or assign 'Unknown' if missing
        patient_points = generate_random_points(gp_location, radius_km, patients_per_gp)  # Generate patient locations
        for lat, lon in patient_points:
            # Append each patient's details, including a unique ID
            patient_data.append({
                "GP_Name": gp_name,
                "Patient_ID": f"Patient_{patient_counter}",  # Generate a unique patient ID
                "Latitude": lat,
                "Longitude": lon
            })
            patient_counter += 1  # Increment the patient ID counter
    
    # Convert the patient data list into a DataFrame
    return pd.DataFrame(patient_data)

def main():
    """
    Main function to process GP surgeries and generate patient data.
    Saves the processed data to CSV files for further use.
    """
    # Step 1: Process GP surgery data from the GeoJSON file
    gp_df = process_geojson(GEOJSON_FILE)  # Extract GP details
    gp_df.to_csv(GP_SURGERIES_CSV, index=False)  # Save processed GP data to a CSV file
    print(f"GP surgeries saved to {GP_SURGERIES_CSV}")

    # Step 2: Generate synthetic patient data around each GP surgery
    patient_df = generate_patient_data(gp_df)  # Generate patient data
    patient_df.to_csv(PATIENT_LOCATIONS_CSV, index=False)  # Save patient data to a CSV file
    print(f"Patient locations saved to {PATIENT_LOCATIONS_CSV}")

if __name__ == "__main__":
    main()  # Run the main function
