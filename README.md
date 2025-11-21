# Particle Swarm Optimization: Bird Nest Finder

A Python implementation of Particle Swarm Optimization (PSO) algorithm that simulates birds finding an optimal nest location while avoiding all predators and staying close to all food sources.

## Overview

This project demonstrates PSO through an analogy where a swarm of birds collectively searches for the best nesting location. The optimal location maximizes distance from all predators while minimizing distance to all food sources.

## Features

- **Interactive PSO Animation**: Real-time visualization of particle movement and convergence
- **Exhaustive Search Comparison**: Compare PSO efficiency against brute-force search
- **Early Convergence Detection**: Automatically stops when solution stabilizes
- **Customizable Parameters**: Adjust map size, particle count, and iterations
- **Fitness Landscape Visualization**: Heatmap showing the complete solution space
- **Performance Metrics**: Detailed comparison of computation time, evaluations, and accuracy

## Requirements

- Python 3.7+
- NumPy
- Matplotlib

## Installation

```bash
pip install numpy matplotlib
```

## Usage

Run the main script:

```bash
python PSO_c.py
```

### Interactive Configuration

The program will prompt you for:

1. **Map Size**:

   - Small (200×200) - 4 predators, 4 food sources
   - Large (800×800) - 8 predators, 8 food sources

2. **Number of Particles**: Number of birds in the swarm (default: 20)

3. **Iterations**: Maximum optimization iterations (default: 50)

### Example Session

```
Choose map size (1 or 2, default=1): 1
Number of birds/particles (default=20): 20
Number of iterations (default=50): 50
```

## Algorithm Details

### PSO Parameters

- **Inertia Weight (w)**: 0.5 - Controls momentum of particles
- **Cognitive Weight (c1)**: 1.5 - Attraction to personal best position
- **Social Weight (c2)**: 1.5 - Attraction to global best position

### Fitness Function

```
Fitness = Distance to Closest Predator - Distance to Furthest Food
```

Higher fitness values indicate better nest locations.

### Velocity Update

```
v(t+1) = w·v(t) + c1·r1·(pbest - x(t)) + c2·r2·(gbest - x(t))
```

Where:

- `v(t)` = current velocity
- `x(t)` = current position
- `pbest` = personal best position
- `gbest` = global best position
- `r1, r2` = random values [0,1]

## Visualizations

### 1. PSO Animation

- **Blue dots**: Individual birds (particles)
- **Blue arrows**: Velocity vectors showing movement direction
- **Yellow star**: Current best nest location
- **Red X**: Predators to avoid
- **Green squares**: Food sources

### 2. Fitness Landscape

- **Heatmap**: Color-coded fitness values across entire map
- **Yellow star**: PSO solution
- **Cyan star**: Optimal solution from exhaustive search
- Shows relative performance and accuracy

### 3. Comparison Statistics

- Computation time comparison
- Evaluation efficiency
- Position accuracy
- Speedup metrics

## Performance

Typical results on a 200×200 map:

- **PSO Time**: ~0.1-0.3 seconds
- **Exhaustive Search Time**: ~2-5 seconds
- **Speedup**: 10-50× faster
- **Evaluation Efficiency**: 100× fewer evaluations
- **Position Accuracy**: 85-99%

## Convergence

The algorithm includes early stopping when:

- Average particle movement < 0.01 (threshold)
- Condition maintained for 5 consecutive iterations

This prevents unnecessary computation once the swarm has converged.

## Code Structure

```
PSO_c.py
├── BirdNestPSO Class
│   ├── __init__()           # Initialize parameters and positions
│   ├── _generate_spaced_points()  # Smart placement of predators/food
│   ├── _fitness()           # Evaluate nest location quality
│   ├── update()             # PSO velocity and position updates
│   ├── exhaustive_search()  # Brute-force optimal solution
│   └── run()                # Main optimization loop
├── visualize_pso()          # Animate PSO process
├── visualize_exhaustive_search()  # Show fitness landscape
├── visualize_comparison()   # Display performance metrics
└── main()                   # User interface and workflow
```

## Customization

### Adjust PSO Parameters

Modify the `BirdNestPSO.__init__()` method:

```python
self.w = 0.5   # Inertia: higher = more exploration
self.c1 = 1.5  # Cognitive: higher = stronger memory
self.c2 = 1.5  # Social: higher = stronger swarm effect
```

### Change Convergence Criteria

```python
self.convergence_threshold = 0.01  # Movement threshold
self.convergence_iterations = 5     # Stability duration
```

### Modify Fitness Function

Edit the `_fitness()` method to implement different objectives:

```python
def _fitness(self, position):
    # Your custom fitness calculation
    return fitness_value
```

## Educational Value

This project demonstrates:

1. **Swarm Intelligence**: Emergent collective behavior
2. **Metaheuristic Algorithms**: Global optimization techniques
3. **Trade-offs**: Speed vs. accuracy in optimization
4. **Visualization**: Understanding algorithm behavior
5. **Performance Analysis**: Benchmarking optimization methods

## Limitations

- PSO may converge to local optima (not guaranteed global optimum)
- Performance depends on parameter tuning
- Best for continuous optimization problems
- Requires fitness function evaluation at each particle position

## License

This project is open source and available for educational purposes.

## Author

Created as an educational demonstration of Particle Swarm Optimization algorithms.

---

**Note**: Results may vary between runs due to random initialization. Multiple runs can help assess algorithm robustness.
