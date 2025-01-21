import numpy as np
from scipy.interpolate import splprep, splev
from scipy.integrate import quad
from numpy.linalg import norm
import math
from angle_measurer_from_graph import generation_number_generator


# Helper function: Compute arc length
def compute_arc_length(tck, t0, t1):
	def velocity_magnitude(t):
		derivative = np.array(splev(t, tck, der=1))
		return norm(derivative)
	arc_length, _ = quad(velocity_magnitude, t0, t1)
	return arc_length

# Helper function: Compute angle between two vectors
def angle_between_vectors(v1, v2):
	dot_product = np.dot(v1, v2)
	norms = norm(v1) * norm(v2)
	return math.acos(np.clip(dot_product / norms, -1.0, 1.0)) * (180 / np.pi)

def acute_angle_between_vectors(v1, v2):
	angle = angle_between_vectors(v1, v2)
	return angle if angle <= 90 else 180 - angle

def compute_mean_curvature_per_unit_length(tck, t_vals):
	total_curvature = 0
	total_arc_length = 0
	for i in range(len(t_vals) - 1):
		t_mid = (t_vals[i] + t_vals[i + 1]) / 2
		c1 = np.array(splev(t_mid, tck, der=1))  # Velocity
		c2 = np.array(splev(t_mid, tck, der=2))  # Acceleration
		numerator = norm(np.cross(c1, c2))
		denominator = norm(c1)**3
		curvature = numerator / denominator if denominator != 0 else 0

		# Weight by the local arc length
		local_arc_length = norm(np.array(splev(t_vals[i + 1], tck, der=1))) * (t_vals[i + 1] - t_vals[i])
		total_curvature += curvature * local_arc_length
		total_arc_length += local_arc_length

	return total_curvature / total_arc_length if total_arc_length != 0 else 0




def analyze_tree_structure_with_generational_results(points, adjacency, gen_nums):
	N = len(points)
	
	# Initialize results dictionaries
	results = {
		"arc_length": {},
		"local_angle": {},
		"global_angle": {},
		"mean_curvature": {},
		"local_dihedral_angle": {},
		"global_dihedral_angle": {}
	}
	local_normals_at_bifurcations = {}
	global_normals_at_bifurcations = {}

	# Identify bifurcation points and endpoints
	degree = adjacency.sum(axis=1)
	bifurcation_indices = np.where(degree > 2)[0]
	endpoint_indices = np.where(degree == 1)[0]

	visited_edges = set()

	def traverse_segment(start, neighbor):
		path = [start]
		current = neighbor
		while current not in bifurcation_indices and current not in endpoint_indices:
			path.append(current)
			next_neighbors = [n for n in np.where(adjacency[current] == 1)[0] if (current, n) not in visited_edges and (n, current) not in visited_edges]
			if not next_neighbors:
				break
			next_node = next_neighbors[0]
			visited_edges.add((current, next_node))
			visited_edges.add((next_node, current))
			current = next_node
		path.append(current)
		return path

	for bifurcation in bifurcation_indices:
		local_vectors = []
		connected_bifurcations = []
		global_vectors = []
		k = 0
		for neighbor in np.where(adjacency[bifurcation] == 1)[0]:
			if (bifurcation, neighbor) not in visited_edges and generation_numbers[bifurcation] == generation_numbers[neighbor]:
				visited_edges.add((bifurcation, neighbor))
				visited_edges.add((neighbor, bifurcation))
				segment = traverse_segment(bifurcation, neighbor)
				segment_points = points[segment]

				# Fit spline
				tck, u = splprep(segment_points.T, s=0, k=3)
				arc_length = compute_arc_length(tck, u[0], u[-1])
				t_vals = np.linspace(u[0], u[-1], 100)
				mean_curvature = compute_mean_curvature_per_unit_length(tck, t_vals)

				# Local vector (from bifurcation to first point in segment)
				tangent_vector = np.array(splev(u[0], tck, der=1))
				local_vector = tangent_vector / np.linalg.norm(tangent_vector)  # Normalize
				local_vectors.append(local_vector)

				# Global vector (from bifurcation to next bifurcation/end)
				global_vector = segment_points[-1] - segment_points[0]
				global_vectors.append(global_vector)

				# Determine the generation of the segment
				generation = gen_nums[segment[0]]

				# Add results to dictionaries
				results["arc_length"].setdefault(generation, []).append(arc_length)
				results["mean_curvature"].setdefault(generation, []).append(mean_curvature)

						# Compute normal vector at bifurcation
				if len(global_vectors) == 2:
					local_normals_at_bifurcations[bifurcation] = np.cross(local_vectors[0], local_vectors[1])
					global_normals_at_bifurcations[bifurcation] = np.cross(global_vectors[0], global_vectors[1])
					results["global_angle"].setdefault(generation, []).append(angle_between_vectors(global_vectors[0], global_vectors[1]))
					results["local_angle"].setdefault(generation, []).append(angle_between_vectors(local_vectors[0], local_vectors[1]))


				# Check if the segment connects to another bifurcation
				if segment[-1] in bifurcation_indices:
					connected_bifurcations.append(segment[-1])

			

		# Compute dihedral angles for connected bifurcations
		for connected_bifurcation in connected_bifurcations:
			if connected_bifurcation in local_normals_at_bifurcations:
				normal1 = local_normals_at_bifurcations[bifurcation]
				normal2 = local_normals_at_bifurcations[connected_bifurcation]
				local_dihedral_angle = acute_angle_between_vectors(normal1, normal2)
				generation = gen_nums[bifurcation]
				results["local_dihedral_angle"].setdefault(generation, []).append(local_dihedral_angle)
			if connected_bifurcation in global_normals_at_bifurcations:
				normal1 = global_normals_at_bifurcations[bifurcation]
				normal2 = global_normals_at_bifurcations[connected_bifurcation]
				global_dihedral_angle = acute_angle_between_vectors(normal1, normal2)
				generation = gen_nums[bifurcation]
				results["global_dihedral_angle"].setdefault(generation, []).append(global_dihedral_angle)
		
	for key in results:
		results[key] = dict(sorted(results[key].items()))

	return results






folder = 'master_thesis_animations/growth_wnt_alpha_bifurcation/isotropic_alpha_wnt_cells/div_time=12000_lam3=0.1_mesbias_relax/adjacency_matrix_and_points_div_time=12000_lam3=0.1_1_scene=6.npy'
data = np.load(folder, allow_pickle=True).item()
adjacency_matrix = data['adjacency_matrix_6']
points = data['points_6']
x, p = data['x_6'], data['p_6']

generation_numbers = generation_number_generator(adjacency_matrix, points, x, p)

results = analyze_tree_structure_with_generational_results(points, adjacency_matrix, generation_numbers)

# Display results
for key, value in results.items():
	print(f"{key}: {value}")