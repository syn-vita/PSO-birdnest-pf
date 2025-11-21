"""
Particle Swarm Optimization Algorithm Implementation
Core PSO logic for finding optimal bird nest location
"""

import numpy as np
from utils import generate_spaced_points


class BirdNestPSO:
    """
    PSO implementation for finding optimal nest location
    Balances distance from predators and proximity to food sources
    """
    
    def __init__(self, map_size=200, n_particles=20, n_predators=4, n_food=4, 
                 fixed_predators=None, fixed_food=None):
        """
        Initialize PSO with map and particle parameters
        
        Args:
            map_size: Size of the square map
            n_particles: Number of particles (birds) in swarm
            n_predators: Number of predator locations
            n_food: Number of food source locations
            fixed_predators: Optional numpy array of fixed predator positions
            fixed_food: Optional numpy array of fixed food positions
        """
        self.map_size = map_size
        self.n_particles = n_particles
        self.n_predators = n_predators
        self.n_food = n_food
        
        # PSO parameters
        self.w = 0.5   # inertia weight
        self.c1 = 1.5  # cognitive (personal best) weight
        self.c2 = 1.5  # social (global best) weight
        
        # Initialize predators and food sources
        if fixed_predators is not None:
            self.predators = fixed_predators
        else:
            self.predators = generate_spaced_points(n_predators, map_size)
        
        if fixed_food is not None:
            self.food_sources = fixed_food
        else:
            self.food_sources = generate_spaced_points(n_food, map_size)
        
        # Initialize particles randomly
        self.particles = np.random.rand(n_particles, 2) * map_size
        self.velocities = np.random.randn(n_particles, 2) * 2
        
        # Personal best positions and fitness
        self.personal_best = self.particles.copy()
        self.personal_best_fitness = np.array([self._fitness(p) for p in self.particles])
        
        # Global best (highest fitness)
        best_idx = np.argmax(self.personal_best_fitness)
        self.global_best = self.personal_best[best_idx].copy()
        self.global_best_fitness = self.personal_best_fitness[best_idx]
        
        # History for visualization
        self.history = []
        
        # Convergence tracking (Make threshold smaller (0.01) and iterations larger (5) for slower convergence stop)
        self.convergence_threshold = 0.2  # Stop if average movement < this
        self.convergence_iterations = 3   # Must be stable for this many iterations
        self.movement_history = []
    
    def _fitness(self, position):
        """
        Calculate fitness of a position
        Fitness = Distance to closest predator - Distance to furthest food
        Higher fitness is better
        
        Args:
            position: numpy array [x, y]
        
        Returns:
            float: fitness value
        """
        # Distance to all predators
        predator_distances = [np.linalg.norm(position - pred) for pred in self.predators]
        closest_predator_dist = min(predator_distances)
        
        # Distance to all food sources
        food_distances = [np.linalg.norm(position - food) for food in self.food_sources]
        furthest_food_dist = max(food_distances)
        
        fitness = closest_predator_dist - furthest_food_dist
        return fitness
    
    def update(self):
        """
        Update particle positions and velocities for one iteration
        
        Returns:
            float: average movement distance of all particles
        """
        # Track positions before update
        old_positions = self.particles.copy()
        
        for i in range(self.n_particles):
            # Update velocity using PSO formula
            r1, r2 = np.random.rand(2)
            
            # Inertia component
            inertia = self.w * self.velocities[i]
            
            # Cognitive component (personal best)
            cognitive = self.c1 * r1 * (self.personal_best[i] - self.particles[i])
            
            # Social component (global best)
            social = self.c2 * r2 * (self.global_best - self.particles[i])
            
            # New velocity
            self.velocities[i] = inertia + cognitive + social
            
            # Limit velocity to prevent overshooting
            speed = np.linalg.norm(self.velocities[i])
            max_speed = self.map_size / 10
            if speed > max_speed:
                self.velocities[i] = (self.velocities[i] / speed) * max_speed
            
            # Update position
            self.particles[i] += self.velocities[i]
            
            # Boundary conditions (bounce back)
            for dim in range(2):
                if self.particles[i][dim] < 0:
                    self.particles[i][dim] = 0
                    self.velocities[i][dim] *= -0.5
                elif self.particles[i][dim] > self.map_size:
                    self.particles[i][dim] = self.map_size
                    self.velocities[i][dim] *= -0.5
            
            # Update personal best
            current_fitness = self._fitness(self.particles[i])
            if current_fitness > self.personal_best_fitness[i]:
                self.personal_best[i] = self.particles[i].copy()
                self.personal_best_fitness[i] = current_fitness
                
                # Update global best
                if current_fitness > self.global_best_fitness:
                    self.global_best = self.particles[i].copy()
                    self.global_best_fitness = current_fitness
        
        # Calculate average movement
        movements = np.linalg.norm(self.particles - old_positions, axis=1)
        avg_movement = np.mean(movements)
        self.movement_history.append(avg_movement)
        
        # Store history for visualization
        self.history.append({
            'particles': self.particles.copy(),
            'global_best': self.global_best.copy(),
            'global_best_fitness': self.global_best_fitness
        })
        
        return avg_movement
    
    def run(self, n_iterations=50):
        """
        Run PSO algorithm with early stopping
        
        Args:
            n_iterations: Maximum number of iterations
        
        Returns:
            tuple: (global_best_position, global_best_fitness, actual_iterations)
        """
        for iteration in range(n_iterations):
            avg_movement = self.update()
            
            # Check for convergence after minimum iterations
            if iteration >= self.convergence_iterations:
                recent_movements = self.movement_history[-self.convergence_iterations:]
                if all(m < self.convergence_threshold for m in recent_movements):
                    print(f"  Convergence detected at iteration {iteration + 1}")
                    print(f"  Average movement: {avg_movement:.4f} (threshold: {self.convergence_threshold})")
                    print(f"  Recent movements: {[f'{m:.4f}' for m in recent_movements]}")
                    return self.global_best, self.global_best_fitness, iteration + 1
        
        print(f"  Completed all {n_iterations} iterations without convergence")
        print(f"  Final average movement: {avg_movement:.4f}")
        return self.global_best, self.global_best_fitness, n_iterations
    
    # Exhaustive search implementation for comparison
    def exhaustive_search(self, resolution=50):
        """
        Perform exhaustive grid search for comparison
        
        Args:
            resolution: Grid resolution (resolution x resolution points)
        
        Returns:
            tuple: (X meshgrid, Y meshgrid, fitness_map)
        """
        x = np.linspace(0, self.map_size, resolution)
        y = np.linspace(0, self.map_size, resolution)
        X, Y = np.meshgrid(x, y)
        
        fitness_map = np.zeros((resolution, resolution))
        
        for i in range(resolution):
            for j in range(resolution):
                position = np.array([X[i, j], Y[i, j]])
                fitness_map[i, j] = self._fitness(position)
        
        return X, Y, fitness_map
