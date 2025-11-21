"""
Utility functions for PSO Bird Nest Finder
Contains helper functions for point generation and other utilities
"""

import numpy as np


def generate_spaced_points(n_points, map_size):
    """
    Generate points with good spacing to avoid clustering
    
    Args:
        n_points: Number of points to generate
        map_size: Size of the map (assumed square)
    
    Returns:
        numpy array of shape (n_points, 2) with generated points
    """
    points = []
    min_distance = map_size / (2 * np.sqrt(n_points) + 1)
    
    attempts = 0
    max_attempts = 1000
    
    while len(points) < n_points and attempts < max_attempts:
        new_point = np.random.rand(2) * map_size
        
        # Check distance from existing points
        if len(points) == 0:
            points.append(new_point)
        else:
            distances = [np.linalg.norm(new_point - p) for p in points]
            if min(distances) > min_distance:
                points.append(new_point)
        
        attempts += 1
    
    # If we couldn't generate enough points, fill the rest randomly
    while len(points) < n_points:
        points.append(np.random.rand(2) * map_size)
    
    return np.array(points)
