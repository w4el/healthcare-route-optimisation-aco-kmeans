# Healthcare Delivery Route Optimisation

> A hybrid metaheuristic algorithm combining K-Means clustering and Ant Colony Optimisation (ACO) to solve a complex, real-world Vehicle Routing Problem (VRP) for healthcare logistics.

This project, part of my Natural Computing module, designs and implements a two-stage solution to optimise routes for healthcare services (e.g., GP home visits), aiming to minimise total travel distance and operational costs.

## 1. Overview

The challenge is a variant of the complex, NP-hard Travelling Salesman Problem (TSP). A brute-force approach is computationally infeasible. This solution intelligently decomposes the problem:

1.  **Clustering:** First, K-Means Clustering is used to partition all patient locations into distinct, localised clusters, assigning each cluster to a specific GP surgery.
2.  **Optimisation:** Second, the Ant Colony Optimisation (ACO) metaheuristic is applied to each cluster independently to find the shortest possible route for that GP to visit all their assigned patients.

## 2. Key Features

* **Hybrid Algorithm:** Implements an innovative two-stage solution, combining an unsupervised clustering algorithm (K-Means) with a powerful metaheuristic (ACO).
* **Complex Problem-Solving:** Solves a real-world, NP-hard optimisation problem (a TSP/VRP variant).
* **Parameter Tuning:** Includes analysis of ACO parameters (e.g., heuristic weighting 'beta', pheromone evaporation 'rho') to find the optimal balance between exploration and exploitation, ensuring a high-quality solution.
* **Data Visualisation:** Uses the Folium library to plot and visualise the final, optimised patient routes on an interactive map.

## 3. Technologies & Libraries Used

* **Core:** Python
* **Data Science:** Pandas, NumPy, Scikit-learn (for `KMeans`)
* **Optimisation:** `aco-learn` (or a similar Ant Colony Optimisation library)
* **Visualisation:** Matplotlib (for convergence plots), Folium (for map visualisation)

## 4. Installation & Usage

1.  Clone the repository:
    `git clone https://github.com/w4el/healthcare-route-optimisation-aco-kmeans.git`
2.  Navigate to the project directory:
    `cd healthcare-route-optimisation-aco-kmeans`
3.  Install the required dependencies (you must create this `requirements.txt` file):
    `pip install -r requirements.txt`

To run the main analysis and see the final route visualisations, open and run the Jupyter Notebook:
`jupyter notebook Optimisation_Analysis.ipynb`

## 5. Results

The hybrid approach demonstrated a significant reduction in total travel distance compared to a non-clustered baseline. The convergence analysis showed that the ACO algorithm effectively found a near-optimal route within 150-200 iterations.
