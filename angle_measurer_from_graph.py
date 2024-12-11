import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from vispy import app, color, scene
from vispy.scene import visuals


def load_tree(path, scene=-1, cell_type=True):
	data = np.load(path, allow_pickle=True)
	x, p = data[:2]
	if len(data) > 2:
		cell_type = data[3]
		tree_scene_x, tree_scene_p, tree_scene_cell_type = x[scene], p[scene], cell_type[scene]
		mask = tree_scene_cell_type != 2
		x = tree_scene_x[mask]
		p = tree_scene_p[mask]
	return x, p

# label each point with a generation number according to number of bifurcation points away from the stem
# the stem should be generation 0

def generation_number_generator(adjacency_matrix, point_array, x, p):
	# Stem initialization
	stem_indices = [34, 676]  # Add more stem indices if needed
	stem_points = set()

	# Identify closest points in point_array for each stem index
	for idx in stem_indices:
		stem_points.add(np.argmin(np.linalg.norm(point_array - (x[idx] - p[idx]), axis=1)))

	# Initialize generation numbers and visited set
	num_points = point_array.shape[0]
	generation_numbers = [-1] * num_points  # -1 means unvisited
	visited = set()

	# Initialize BFS for each stem point
	queues = {point: deque([(point, 0)]) for point in stem_points}
	for point in stem_points:
		generation_numbers[point] = 0
		visited.add(point)

	# Set to track bifurcation points
	bifurcation_points = set()

	# Label all stem points as generation 0
	while any(queues.values()):  # While at least one queue is not empty
		for point, queue in list(queues.items()):
			if not queue:
				continue

			current_point, current_generation = queue.popleft()
			neighbors = np.where(adjacency_matrix[current_point] == 1)[0]

			for neighbor in neighbors:
				if neighbor in visited:
					continue

				visited.add(neighbor)

				# Check if the neighbor is a bifurcation point
				neighbor_connections = np.where(adjacency_matrix[neighbor] == 1)[0]
				if len(neighbor_connections) >= 3:
					bifurcation_points.add(neighbor)
				else:
					generation_numbers[neighbor] = 0
					queue.append((neighbor, current_generation))

			# Stop propagating once this path has reached a bifurcation
			if neighbor in bifurcation_points:
				queue.clear()

	# Continue labeling beyond bifurcation points
	queue = deque([(bif_point, 1) for bif_point in bifurcation_points])
	for bif_point in bifurcation_points:
		generation_numbers[bif_point] = 1
		visited.add(bif_point)

	while queue:
		current_point, current_generation = queue.popleft()
		neighbors = np.where(adjacency_matrix[current_point] == 1)[0]

		for neighbor in neighbors:
			if neighbor in visited:
				continue

			visited.add(neighbor)

			# Check if the neighbor is a bifurcation point
			neighbor_connections = np.where(adjacency_matrix[neighbor] == 1)[0]
			if len(neighbor_connections) >= 3:
				next_generation = current_generation + 1
			else:
				next_generation = current_generation

			generation_numbers[neighbor] = next_generation
			queue.append((neighbor, next_generation))

	return np.array(generation_numbers)

# create a function that takes adjacency matrix and point array as input and returns the angles between the edges
def local_angle_measurer(adjacency_matrix, point_array, generation_numbers):
	# find the bifurcation points and immediately adjacent points
	bifurcation_points = np.where(np.sum(adjacency_matrix, axis=1) >= 3)[0]
	bifurcation_neighbors = {bif_point: np.where(adjacency_matrix[bif_point] == 1)[0] for bif_point in bifurcation_points}
	# we want to measure the angle between the two neighbors with the larger generation number with respect to the vector from the smaller generation to the larger generation    
	local_angles = {}
	for bif_point_index, neighbors in bifurcation_neighbors.items():
		# find the generation numbers of the neighbors
		neighbor_generations = generation_numbers[neighbors]
		# find the indices of the neighbors with the largest generation number
		indices_full = np.argsort(neighbor_generations)
		indices = indices_full[-2:]
		indices_min = indices_full[0]
		if 0 in neighbor_generations[indices]:
			continue
		# find the points of the neighbors with the largest generation number
		neighbor_points = point_array[neighbors[indices]]
		# find the point of the bifurcation point
		bif_point = point_array[bif_point_index]
		# find the vectors between the bifurcation point and the neighbors
		vectors = neighbor_points - bif_point
		# find the angle between the vectors
		local_angle = np.arccos(np.dot(vectors[0], vectors[1]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])))
		# convert to degrees
		local_angle = np.degrees(local_angle)
		# in the local_angles dictionary, the key is the lowest generation number involved and the value is the angle
		# there will be more than one angle for each generation number, append them all for each generation number
		if neighbor_generations[indices_min] in local_angles:
			local_angles[neighbor_generations[indices_min]].append(local_angle)
		else:
			local_angles[neighbor_generations[indices_min]] = [local_angle]      
		# store the vectors in the local_vectors dictionary

	# sort the dictionary so that the keys are in ascending order
	local_angles = dict(sorted(local_angles.items()))







	return local_angles


# create function that takes adjacency matrix and point array as input and returns the length of each segment of the graph (from bifurcationpoint to bifurcationpoint) indexed by the lower generation number

def length_measurer(adjacency_matrix, point_array, generation_numbers):
	# Identify bifurcation points and terminal points (exclude generation 0 for terminal points)
	bifurcation_points = np.where(np.sum(adjacency_matrix, axis=1) >= 3)[0]
	terminal_points = np.where((np.sum(adjacency_matrix, axis=1) == 1) & (generation_numbers > 0))[0]
	key_points = set(bifurcation_points).union(set(terminal_points))
	# print amount of bifurcation points with generation number 1
	
	# Initialize dictionary to store lengths indexed by lower generation number
	bifurcation_connections = []
	segment_lengths = {}
	global_vectors = {}
	global_angles = {}
	global_dihedral_normals = {}
	global_dihedral_angles = {}
	local_vectors = {}
	local_angles = {}
	local_dihedral_normals = {}
	local_dihedral_angles = {}





	# Traverse the graph
	visited = set()
	# order bifurcation points by generation number
	bifurcation_points = sorted(bifurcation_points, key=lambda x: generation_numbers[x])


	for start_point in bifurcation_points:
		if start_point in visited:
			continue
		
		# BFS to find paths between key points
		queue = [(start_point, 0)]  # (current point, cumulative length)
		visited.add(start_point)
		
		while queue:
			current_point, current_length = queue.pop(0)
			neighbors = np.where(adjacency_matrix[current_point] == 1)[0]
			
			for neighbor in neighbors:
				# Skip already visited points not in the bifurcation list or points in the stem
				if neighbor in visited and neighbor not in bifurcation_points:
					continue
				if generation_numbers[neighbor] == 0:
					continue
				# Add distance to current length
				distance = np.linalg.norm(point_array[current_point] - point_array[neighbor])
				new_length = current_length + distance

				if current_point == start_point:
					# if there are any zeros in neighbors, then dont continue with this if statement
					if 0 in generation_numbers[neighbors]:
						continue
					from_gen, to_gen = generation_numbers[start_point], generation_numbers[neighbor] 
					if from_gen == to_gen: # discard segments between points of the same generation
						vector = point_array[neighbor] - point_array[start_point]
						if current_point not in local_vectors:
							local_vectors[current_point] = []
						local_vectors[current_point].append(vector)


				# If the neighbor is a key point, record the segment length
				if neighbor in key_points:
					from_gen, to_gen = generation_numbers[start_point], generation_numbers[neighbor]
					if from_gen == to_gen: # discard segments between points of the same generation
						if neighbor in terminal_points:
							pass
						else:
							continue
					lower_generation = min(from_gen, to_gen)
					if lower_generation not in segment_lengths:
						segment_lengths[lower_generation] = []
					segment_lengths[lower_generation].append(new_length)

					if lower_generation<2:
						continue
					vector = point_array[neighbor] - point_array[start_point]
					
					if from_gen > to_gen:
						vector = -vector	
						if neighbor not in global_vectors:
							global_vectors[neighbor] = []
						global_vectors[neighbor].append(vector)
						if start_point in terminal_points:
							continue
						bifurcation_connections.append((neighbor, start_point))
					else:
						if start_point not in global_vectors:
							global_vectors[start_point] = []
						global_vectors[start_point].append(vector)
						if neighbor in terminal_points:
							continue
						bifurcation_connections.append((start_point, neighbor))
					#visited.add(neighbor)
				else:
					# Continue traversal
					queue.append((neighbor, new_length))
					visited.add(neighbor)
	# Now we have the global vectors indexed by the starting bifurcation point
	# We want to find the angle between the vectors
	for key, value in global_vectors.items():
		if len(value) > 1:
			vector1 = value[0]
			vector2 = value[1]
			angle = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
			angle = np.degrees(angle)
			generation_number = generation_numbers[key]
			if generation_number not in global_angles:
				global_angles[generation_number] = []
			global_angles[generation_number].append(angle)
		else:
			global_vectors[key] = 0
			print(f'No global vector for bifurcation point {key}')

	# Now we have the local vectors indexed by the terminal point
	# We want to find the angle between the vectors
	for key, value in local_vectors.items():
		if len(value) > 1:
			vector1 = value[0]
			vector2 = value[1]
			angle = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
			angle = np.degrees(angle)
			print(angle)
			generation_number = generation_numbers[key]
			if generation_number not in local_angles:
				local_angles[generation_number] = []
			local_angles[generation_number].append(angle)
		else:
			local_vectors[key] = 0
			print(f'No local vector for terminal point {key}')
	
	# Now we have the indices of connected bifurcation points (or end points). We want to find the dihedral angle between the planes spanned by the local/global vectors of successive bifurcations
	for connection in bifurcation_connections:
		# retrieve the local vectors of the connected bifurcation points
		bif1, bif2 = connection
		local_vector1_1 = local_vectors[bif1][0]
		local_vector1_2 = local_vectors[bif1][1]
		local_vector2_1 = local_vectors[bif2][0]
		local_vector2_2 = local_vectors[bif2][1]
		# retrieve the global vectors of the connected bifurcation points
		global_vector1_1 = global_vectors[bif1][0]
		global_vector1_2 = global_vectors[bif1][1]
		global_vector2_1 = global_vectors[bif2][0]
		global_vector2_2 = global_vectors[bif2][1]
		# find the normal vectors of the planes spanned by the local vectors
		local_normal1 = np.cross(local_vector1_1, local_vector1_2)
		local_normal2 = np.cross(local_vector2_1, local_vector2_2)
		# find the normal vectors of the planes spanned by the global vectors
		global_normal1 = np.cross(global_vector1_1, global_vector1_2)
		global_normal2 = np.cross(global_vector2_1, global_vector2_2)
		# find the dihedral angle between the planes spanned by the local vectors
		local_dihedral_angle = np.arccos(np.abs(np.dot(local_normal1, local_normal2)) / (np.linalg.norm(local_normal1) * np.linalg.norm(local_normal2)))
		local_dihedral_angle = np.degrees(local_dihedral_angle)
		# find the dihedral angle between the planes spanned by the global vectors
		global_dihedral_angle = np.arccos(np.abs(np.dot(global_normal1, global_normal2)) / (np.linalg.norm(global_normal1) * np.linalg.norm(global_normal2)))
		global_dihedral_angle = np.degrees(global_dihedral_angle)
		# store the dihedral angles in the respective dictionaries indexed by the lower generation number
		generation_number = min(generation_numbers[bif1], generation_numbers[bif2])

		if bif1 not in local_dihedral_normals:
			local_dihedral_normals[bif1] = []
		local_dihedral_normals[bif1].append(local_normal1)

		if bif2 not in local_dihedral_normals:
			local_dihedral_normals[bif2] = []
		local_dihedral_normals[bif2].append(local_normal2)

		if bif1 not in global_dihedral_normals:
			global_dihedral_normals[bif1] = []
		global_dihedral_normals[bif1].append(global_normal1)

		if bif2 not in global_dihedral_normals:
			global_dihedral_normals[bif2] = []
		global_dihedral_normals[bif2].append(global_normal2)


		if generation_number not in local_dihedral_angles:
			local_dihedral_angles[generation_number] = []
		local_dihedral_angles[generation_number].append(local_dihedral_angle)
		if generation_number not in global_dihedral_angles:
			global_dihedral_angles[generation_number] = []
		global_dihedral_angles[generation_number].append(global_dihedral_angle)



	# Sort the dictionaries so that the keys are in ascending order
	global_angles = dict(sorted(global_angles.items()))
	local_angles = dict(sorted(local_angles.items()))
	segment_lengths = dict(sorted(segment_lengths.items()))
	local_dihedral_angles = dict(sorted(local_dihedral_angles.items()))
	global_dihedral_angles = dict(sorted(global_dihedral_angles.items()))
	

	return segment_lengths, global_angles, global_dihedral_angles, global_vectors, global_dihedral_normals, local_angles, local_dihedral_angles, local_vectors, local_dihedral_normals




div_time = 11000
lam3 = 0.1
path = f'master_thesis_animations/growth_wnt_alpha_bifurcation/isotropic_alpha_wnt_cells/div_time={div_time}_lam3={lam3}/sim_mes_div_time={div_time}_lam3={lam3}.npy'
path2 = f'master_thesis_animations/growth_wnt_alpha_bifurcation/isotropic_alpha_wnt_cells/div_time={div_time}_lam3={lam3}/adjacency_matrix_and_points.npy'
x, p = load_tree(path, scene=-1, cell_type=True)
skeleton_data = np.load(path2, allow_pickle=True).item()
adjacency_matrix, points = skeleton_data['adjacency_matrix'], skeleton_data['points']


generation_numbers = generation_number_generator(adjacency_matrix, points, x, p)

segment_lengths, global_angles, global_dihedral_angles, global_vectors, global_dihedral_normals, local_angles, local_dihedral_angles, local_vectors, local_dihedral_normals = length_measurer(adjacency_matrix, points, generation_numbers)
print(f'global_dihedral_angles: {global_dihedral_angles}')
# plot the global vectors and the original structure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
for key, value in local_dihedral_normals.items():
	for vector in value:
		ax.quiver(points[key][0], points[key][1], points[key][2], vector[0], vector[1], vector[2], length=1)

plt.show()


# Normalize generation numbers for coloring
unique_generations = np.unique(generation_numbers)
# normalize generation numbers to be between 0 and 1
generation_numbers_norm = (generation_numbers - unique_generations.min()) / (unique_generations.max() - unique_generations.min())
num_generations = len(unique_generations)
generation_colors = color.get_colormap('viridis').map(generation_numbers_norm)
#colors = generation_colors(generation_numbers_norm)

'''
# Map each generation to a color
#generation_color_map = {gen: generation_colors[i] for i, gen in enumerate(unique_generations)}
#colors = np.array([generation_color_map[gen] for gen in generation_numbers])
# Create a canvas and a 3D scatter plot
canvas = scene.SceneCanvas(keys='interactive', show=True)

canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

structure = visuals.Markers(spherical=True, scaling=True)
structure.set_data(x, face_color=[1,1,1, 0.4], size=1.2, edge_width=0)
view.add(structure)

# Generate line segments from adjacency matrix
line_segments = []
for i in range(adjacency_matrix.shape[0]):
	for j in range(i + 1, adjacency_matrix.shape[1]):  # Check upper triangle to avoid duplicates
		if adjacency_matrix[i, j] > 0:  # Connection exists
			line_segments.append(points[i])  # Start of line
			line_segments.append(points[j])  # End of line

# Convert to a NumPy array
line_segments = np.array(line_segments)

# Add all lines as a single visual
line_visual = visuals.Line()
line_visual.set_data(line_segments, connect='segments', color='red', width=5)
view.add(line_visual)


# Create the scatter plot
scatter = visuals.Markers(spherical=True, scaling=True)
scatter.set_data(points, face_color=generation_colors, edge_width=0, size=2)
view.add(scatter)
view.camera = 'fly'
view.camera.move_speed = 0.1  # Reduce movement speed


# Run the application
if __name__ == '__main__':
	app.run()
'''