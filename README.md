# Healthcare Delivery Route Optimisation

> A high-performance optimisation solution for solving the complex Vehicle Routing Problem (VRP), a variant of the Travelling Salesman Problem (TSP). This project implements a novel hybrid algorithm combining K-Means clustering with Ant Colony Optimisation (ACO) to find near-optimal routes for healthcare logistics.



## 1. Project Goal & Overview

The challenge of this project was to design a system that could efficiently plan routes for healthcare services (e.g., GP home visits) to minimise total travel distance and operational costs. A brute-force approach to this NP-hard problem is computationally infeasible.

This solution intelligently decomposes the problem into two distinct stages:

1.  **Clustering:** The K-Means algorithm is first applied to partition all patient locations into distinct, geographically-localised clusters. Each cluster represents a manageable service zone.
2.  **Optimisation:** The Ant Colony Optimisation (ACO) metaheuristic is then applied to each cluster independently. The ACO algorithm finds the shortest, most efficient path *within* each zone, creating a complete and optimised route for each service vehicle.

This hybrid approach makes the problem computationally tractable and produces a highly efficient solution.

## 2. Key Features & Technical Implementation

* **Hybrid Algorithmic Model:** Implements an innovative two-stage solution, combining an unsupervised clustering algorithm (K-Means) with a powerful, nature-inspired metaheuristic (ACO).
* **K-Means Clustering:** Leverages `scikit-learn`'s `KMeans` to effectively partition a dataset of synthetic patient coordinates (modelled for the Greenwich area) into logical service zones.
* **Ant Colony Optimisation:** A custom ACO implementation is used to find the shortest path within each cluster. This involves a detailed convergence analysis and parameter tuning process.
* **Parameter Tuning:** The system was optimised by tuning key ACO parameters, such as the heuristic weighting (**beta**) and pheromone evaporation rate (**rho**), to find the optimal balance between solution quality and computational time.
* **Interactive Visualisation:** Uses the **Folium** library to render the final, optimised routes on an interactive map, providing a clear and powerful demonstration of the solution's efficacy.

## 3. Technologies & Libraries Used

* **Core:** Python
* **Data Science:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (for `KMeans`)
* **Optimisation:** `aco-learn` (or similar Ant Colony Optimisation library)
* **Visualisation:** Matplotlib (for convergence plots), Folium (for map rendering)

## 4. Installation & Usage

1.  Clone the repository:
    `git clone https://github.com/w4el/healthcare-route-optimisation-aco-kmeans.git`
2.  Navigate to the project directory:
    `cd healthcare-route-optimisation-aco-kmeans`
3.  Install the required dependencies from the `requirements.txt` file:
    `pip install -r requirements.txt`

To run the main analysis and generate the route visualisations, open and execute the Jupyter Notebook:
`jupyter notebook Optimisation_Analysis.ipynb`

## 5. Results

The integration of K-Means clustering and ACO significantly reduced total travel distances compared to non-clustered baseline approaches. The convergence analysis demonstrated that the ACO algorithm consistently found a near-optimal route within 150-200 iterations, proving the model is both efficient and effective for real-world logistics planning.
