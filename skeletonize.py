import numpy as np
import random
from scipy.spatial import distance
from vispy import scene, color
from vispy.scene import visuals
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as spatial
import vg
import time



np.random.seed(0)


def load_tree(path, scene=-1):
	data = np.load(path, allow_pickle=True)
	x, p, q = data[:3]
	cell_type = data[3]
	epc = data[-1]
	tree_scene_x, tree_scene_p, tree_scene_q, tree_scene_cell_type, tree_scene_epc = x[scene], p[scene], q[scene], cell_type[scene], epc[scene]
	mask = tree_scene_cell_type != 2
	tree_scene_x = tree_scene_x[mask]
	tree_scene_p = tree_scene_p[mask]
	tree_scene_q = tree_scene_q[mask]
	tree_scene_epc = tree_scene_epc[mask]
	return tree_scene_x, tree_scene_p, tree_scene_q, tree_scene_epc



# Function to find the closest point on the line for multiple indices

def closest_point_on_lines(x, p, threshold, indices):

	results = []

	for index in indices:

		# Selected point and polarity
		x0 = x[index]
		p0 = p[index]
		
		# Remove the selected point
		x_other = np.delete(x, index, axis=0)

		# Compute distances
		distances = np.linalg.norm(np.cross(x_other - x0, p0), axis=1)
		
		# Find the index of the minimum distance
		closest_index = np.argmin(distances)
		#closest_distance = distances[closest_index]

		distance_between_points = np.linalg.norm(x_other[closest_index]-x0)#+2*np.dot(-x0 + x_other[closest_index], p0)*p0)

		polarity_position_alignment = np.dot(p0,(x_other[closest_index]-x0)/distance_between_points) 
		polarity_polarity_alignment = np.dot(p0, p[closest_index])
		if (distance_between_points < threshold) & (polarity_position_alignment < 0) & (polarity_polarity_alignment < 0):
			# if the polarities are in the opposite direction keep them
			closest_point = x_other[closest_index]
			results.append((x0, p0, closest_point, closest_index))
		else:
			pass

	midpoints = np.empty((0, 3))

	for (x0, p0, closest_point, closest_index) in results:
		if closest_point is not None:

			# Midpoint
			midpoint = (x0 + closest_point) / 2
			midpoints = np.vstack((midpoints, midpoint))


	return results, midpoints







def thin_line(points, point_cloud_thickness=0.53, iterations=1,sample_points=0):
	if sample_points != 0:
		points = points[:sample_points]
	
	# Sort points into KDTree for nearest neighbors computation later
	point_tree = spatial.cKDTree(points)

	# Empty array for transformed points
	new_points = []
	# Empty array for regression lines corresponding ^^ points
	regression_lines = []
	for point in point_tree.data:
		# Get list of points within specified radius {point_cloud_thickness}
		start_time = time.perf_counter()
		points_in_radius = point_tree.data[point_tree.query_ball_point(point, point_cloud_thickness)]

		# Get mean of points within radius
		start_time = time.perf_counter()
		data_mean = points_in_radius.mean(axis=0)

		# Calulate 3D regression line/principal component in point form with 2 coordinates
		uu, dd, vv = np.linalg.svd(points_in_radius - data_mean)
		linepts = vv[0] * np.mgrid[-1:1:2j][:, np.newaxis]
		linepts += data_mean
		regression_lines.append(list(linepts))

		# Project original point onto 3D regression line
		ap = point - linepts[0]
		ab = linepts[1] - linepts[0]
		point_moved = linepts[0] + np.dot(ap,ab) / np.dot(ab,ab) * ab

		new_points.append(list(point_moved))
	return np.array(new_points), regression_lines




def thin_line(points, point_cloud_thickness=0.53, iterations=1,sample_points=0):
	
	# Iterative thinning of points
	for i in range(iterations):
		if sample_points != 0:
			points = points[:sample_points]
		
		# Sort points into KDTree for nearest neighbors computation later
		point_tree = spatial.cKDTree(points)

		# Empty array for transformed points
		new_points = []
		# Empty array for regression lines corresponding ^^ points
		regression_lines = []
		for point in point_tree.data:
			# Get list of points within specified radius {point_cloud_thickness}
			start_time = time.perf_counter()
			points_in_radius = point_tree.data[point_tree.query_ball_point(point, point_cloud_thickness)]

			# Get mean of points within radius
			start_time = time.perf_counter()
			data_mean = points_in_radius.mean(axis=0)

			# Calulate 3D regression line/principal component in point form with 2 coordinates
			uu, dd, vv = np.linalg.svd(points_in_radius - data_mean)
			linepts = vv[0] * np.mgrid[-1:1:2j][:, np.newaxis]
			linepts += data_mean
			regression_lines.append(list(linepts))

			# Project original point onto 3D regression line
			ap = point - linepts[0]
			ab = linepts[1] - linepts[0]
			point_moved = linepts[0] + np.dot(ap,ab) / np.dot(ab,ab) * ab

			new_points.append(list(point_moved))
		points = np.array(new_points)
	return np.array(new_points), regression_lines


# Sorts points outputed from thin_points()
def sort_points(points, regression_lines, start_index = 522, sorted_point_distance=0.2):
	sort_points_time = time.perf_counter()
	# Index of point to be sorted
	index = start_index

	# sorted points array for left and right of intial point to be sorted
	sort_points_left = [points[index]]
	sort_points_right = []

	# Regression line of previously sorted point
	regression_line_prev = regression_lines[index][1] - regression_lines[index][0]

	# Sort points into KDTree for nearest neighbors computation later
	point_tree = spatial.cKDTree(points)


	# Iterative add points sequentially to the sort_points_left array
	while 1:
		# Calulate regression line vector; makes sure line vector is similar direction as previous regression line
		v = regression_lines[index][1] - regression_lines[index][0]
		if np.dot(regression_line_prev, v ) / (np.linalg.norm(regression_line_prev) * np.linalg.norm(v))  < 0:
			v = regression_lines[index][0] - regression_lines[index][1]
		regression_line_prev = v

		# Find point {distR_point} on regression line distance {sorted_point_distance} from original point 
		distR_point = points[index] + ((v / np.linalg.norm(v)) * sorted_point_distance)

		# Search nearest neighbors of distR_point within radius {sorted_point_distance / 3}
		points_in_radius = point_tree.data[point_tree.query_ball_point(distR_point, sorted_point_distance / 3)]
		if len(points_in_radius) < 1:
			break

		# Neighbor of distR_point with smallest angle to regression line vector is selected as next point in order
		# CAN BE OPTIMIZED
		# 
		nearest_point = points_in_radius[0]
		distR_point_vector = distR_point - points[index]
		nearest_point_vector = nearest_point - points[index]
		for x in points_in_radius: 
			x_vector = x - points[index]
			if vg.angle(distR_point_vector, x_vector) < vg.angle(distR_point_vector, nearest_point_vector):
				nearest_point_vector = nearest_point - points[index]
				nearest_point = x
				
		index = np.where(points == nearest_point)[0][0]

		# Add nearest point to 'sort_points_left' array
		sort_points_left.append(nearest_point)
	# Do it again but in the other direction of initial starting point 
	index = start_index
	regression_line_prev = regression_lines[index][1] - regression_lines[index][0]
	while 1:
		# Calulate regression line vector; makes sure line vector is similar direction as previous regression line
		v = regression_lines[index][1] - regression_lines[index][0]
		if np.dot(regression_line_prev, v ) / (np.linalg.norm(regression_line_prev) * np.linalg.norm(v))  < 0:
			v = regression_lines[index][0] - regression_lines[index][1]
		regression_line_prev = v

		# Find point {distR_point} on regression line distance {sorted_point_distance} from original point 
		# Now vector is substracted from the point to go in other direction
		distR_point = points[index] - ((v / np.linalg.norm(v)) * sorted_point_distance)

		# Search nearest neighbors of distR_point within radius {sorted_point_distance / 3}
		points_in_radius = point_tree.data[point_tree.query_ball_point(distR_point, sorted_point_distance / 3)]
		if len(points_in_radius) < 1:
			break

		# Neighbor of distR_point with smallest angle to regression line vector is selected as next point in order
		# CAN BE OPTIMIZED
		# 
		nearest_point = points_in_radius[0]
		distR_point_vector = distR_point - points[index]
		nearest_point_vector = nearest_point - points[index]
		for x in points_in_radius: 
			x_vector = x - points[index]
			if vg.angle(distR_point_vector, x_vector) < vg.angle(distR_point_vector, nearest_point_vector):
				nearest_point_vector = nearest_point - points[index]
				nearest_point = x
		index = np.where(points == nearest_point)[0][0]

		# Add next point to 'sort_points_right' array
		sort_points_right.append(nearest_point)

	if len(sort_points_left) == 0:
		return np.array(sort_points_right)
	if len(sort_points_right) == 0:
		return np.array(sort_points_left)
	#if len(sort_points_left) == 0 and len(sort_points_right) == 0:
	#	break

	# calculate distance from sort_points_left[-1] to sort_points_right[0] and the distance from sort_points_left[0] to sort_points_right[0]
	print(f'lenght of left: {len(sort_points_left)} and lenght of right: {len(sort_points_right)}')
	dist_left_to_right = np.linalg.norm(sort_points_left[-1] - sort_points_right[0])
	dist_start_to_right = np.linalg.norm(sort_points_left[0] - sort_points_right[0])

	
	# if the distance from sort_points_left[-1] to sort_points_right[0] is smaller than the distance from sort_points_left[0] to sort_points_right[0]
	# combine the two arrays in the order of sort_points_left + sort_points_right
	# else combine the two arrays in the order of sort_points_right + sort_points_left
	if dist_left_to_right < dist_start_to_right:
		sorted_points_combined = sort_points_left + sort_points_right
	else:
		sorted_points_combined = sort_points_left[::-1] + sort_points_right

	print("--- %s seconds to sort points ---" % (time.perf_counter() - sort_points_time))
	return np.array(sorted_points_combined)

# make connected lines in every segment
def sort_points_in_segments(thinned_points, regression_lines, sorted_points_distance=2):
	
	allowed_indices = list(range(len(thinned_points)))

	# collect the sorted points in a list
	sorted_points_collected = []

	while True:
		sorted_points = sort_points(thinned_points[allowed_indices], [regression_lines[i] for i in allowed_indices], start_index = np.random.choice(len(thinned_points[allowed_indices])), sorted_point_distance=sorted_points_distance)
		if len(sorted_points) > 1: # discard lines with only one point
			sorted_points_collected.append(sorted_points)

		# delete all the points in thinned_points that are within sorted_points_distance of the sorted_points
		distances = np.linalg.norm(thinned_points[:, None] - sorted_points, axis=2)
		indices = np.where(np.min(distances, axis=1) < sorted_points_distance)[0]
		allowed_indices = [i for i in allowed_indices if i not in indices]
		print(len(allowed_indices))
		if len(allowed_indices) < 1: # when no points are left we end the loop and return the line collections
			break
	return sorted_points_collected

def create_adjacency_matrix(sorted_points_collected):
	'''
	Takes in the sorted points in a list of arrays format and returns an adjacency matrix where each point is connected to the next point in the array. 
	Endpoints are connected only to the next or previous point in the array. 
	Also return the points array that correspond to the adjacency matrix and the endpoints of the segments
	'''
	# Flatten the list of arrays into a single array of points
	points = np.vstack(sorted_points_collected)
	adjacency_list = {}
	endpoints = []  # To store the indices of the endpoints

	current_index = 0
	for arr in sorted_points_collected:
		num_points = len(arr)

		# Add connections for consecutive points in the current array
		for i in range(num_points - 1):
			if current_index + i not in adjacency_list:
				adjacency_list[current_index + i] = []
			if current_index + i + 1 not in adjacency_list:
				adjacency_list[current_index + i + 1] = []

			# Connect point i with point i+1
			adjacency_list[current_index + i].append(current_index + i + 1)
			adjacency_list[current_index + i + 1].append(current_index + i)

		# Save the endpoints (first and last points of this segment)
		endpoints.append(current_index)  # First point
		endpoints.append(current_index + num_points - 1)  # Last point

		# Update current_index for the next array
		current_index += num_points

	# build the adjacency matrix
	adjacency_matrix = np.zeros((len(points), len(points)))
	for key, neighbors in adjacency_list.items():
		for neighbor in neighbors:
			adjacency_matrix[key, neighbor] = 1
	
	return adjacency_matrix, endpoints, points

# using the adjecancy matrix, visualize the connected points 

def visualize_connected_points(points, adjacency_matrix):
	# Visualize the results
	canvas = scene.SceneCanvas(keys='interactive', show=True)
	view = canvas.central_widget.add_view()

	# Add points to the scene
	original_points = visuals.Markers(spherical=True, scaling=True)
	original_points.set_data(points, face_color=[1,1,1, 0.5], size=1, edge_width=0)
	view.add(original_points)

	# Add lines to the scene
	indices = np.where(adjacency_matrix == 1)
	print(indices[0])
	lines = np.array([points[indices[0]], points[indices[1]]]).T
	line = scene.Line(
		lines,
		color='blue',  # Line color
		width=5.0,     # Line width
		connect='segments',  # Connect points sequentially without closing the loop
		parent=view.scene,
	)
	view.add(line)

	view.camera = 'fly'
	view.camera.move_speed = 0.1  # Reduce movement speed

	## Run the Vispy application
	if __name__ == '__main__':
		canvas.app.run()

	
def calculate_midpoints_of_three_endpoints(endpoints, points, distance_threshold=10):
	triplets = combinations(endpoints, 3)
	valid_triplets = []
	for triplet in triplets:
		coords = points[np.array(triplet)]
		midpoint = np.mean(coords, axis=0)
		distances_to_midpoint = np.linalg.norm(coords - midpoint, axis=1)
		
		# Check if all points in the triplet are sufficiently close to the midpoint
		if np.all(distances_to_midpoint < distance_threshold):
			valid_triplets.append((triplet, midpoint))
	return valid_triplets
	
	


def connect_segments(sorted_points_collected, radius=15):


	# make a flat list of all the points
	points = []
	left_endpoints = []
	right_endpoints = []
	index = 0
	for segment in sorted_points_collected:
		# add the first and last point of the segment to the endpoints along with the index of the endpoints in the points list
		left_endpoints.append(index)
		right_endpoints.append(index + len(segment) - 1)
		index += len(segment)
		for point in segment:
			points.append(point)
	
	# make a KDTree of all the points
	points_tree = spatial.cKDTree(points)
	threshold = 8
	used_endpoints = []
	connections = []

	for left_index in left_endpoints:
		# get the point of the left endpoint
		if left_index in used_endpoints:
			continue
		left_point = points[left_index]

		# get the direction line of the left endpoint
		second_point = points[left_index + 1]
		left_vector = left_point-second_point 		
		left_vector = left_vector / np.linalg.norm(left_vector)

		# get the nearest point to the left endpoint that are not in the same segment
		# get the points in a radius of threshold
		indices = points_tree.query_ball_point(left_point, threshold)
		# sort the indices by distance so smaller indices are checked first
		indices = sorted(indices, key=lambda x: distance.euclidean(left_point, points[x]))
		for idx in indices:
			if idx < left_index or idx > right_endpoints[left_endpoints.index(left_index)]:
				# get the vector from the left point to the nearest point
				direction_vector = points[idx] - left_point
				direction_vector = direction_vector / np.linalg.norm(direction_vector)
				# check if the vectors are parallel
				if np.dot(left_vector, direction_vector) > 0.1:
					# add the connection between the two points
					if idx in left_endpoints or idx in right_endpoints:
						used_endpoints.append(idx)
					connections.append((left_index, idx))
					break

	for right_index in right_endpoints:
		# get the point of the right endpoint
		if right_index in used_endpoints:
			continue
		right_point = points[right_index]

		# get the direction line of the right endpoint
		second_point = points[right_index - 1]
		right_vector = right_point - second_point		
		right_vector = right_vector / np.linalg.norm(right_vector)

		# get the nearest point to the right endpoint that are not in the same segment
		# get the points in a radius of threshold
		indices = points_tree.query_ball_point(right_point, threshold)
		# sort the indices by distance so smaller indices are checked first
		indices = sorted(indices, key=lambda x: distance.euclidean(right_point, points[x]))
		for idx in indices:
			if idx < left_endpoints[right_endpoints.index(right_index)] or idx > right_index:
				# get the vector from the right point to the nearest point
				direction_vector = points[idx] - right_point
				direction_vector = direction_vector / np.linalg.norm(direction_vector)
				# check if the vectors are parallel
				if np.dot(right_vector, direction_vector) > 0.1:
					# add the connection between the two points
					if idx in left_endpoints or idx in right_endpoints:
						used_endpoints.append(idx)
					connections.append((right_index, idx))
					break
		
	# make the adjacency matrix
	adjacency_matrix, endpoints, points = create_adjacency_matrix(sorted_points_collected)
	for connection in connections:
		adjacency_matrix[connection[0], connection[1]] = 1
		adjacency_matrix[connection[1], connection[0]] = 1
		if connection[0] in endpoints:
			endpoints.remove(connection[0])
		if connection[1] in endpoints:
			endpoints.remove(connection[1])
	 
	#number_unpaired_endpoints = len(endpoints)
	#if (number_unpaired_endpoints * (number_unpaired_endpoints - 1) * (number_unpaired_endpoints - 2) / 6) < 10000:
	#	valid_triplets = calculate_midpoints_of_three_endpoints(endpoints, points, distance_threshold=10)
	#	for triplet, midpoint in valid_triplets:
	#		# add the midpoint to the points array
	#		points = np.vstack((points, midpoint))
	#		# add the midpoint to the endpoints
	#		endpoints.append(len(points) - 1)
	#		# connect the midpoint to the three endpoints by adding an extra row and column to the adjacency matrix
	#		adjacency_matrix = np.pad(adjacency_matrix, ((0, 1), (0, 1)), mode='constant')
	#		adjacency_matrix[-1, -1] = 0
	#		for i in range(3):
	#			adjacency_matrix[-1, triplet[i]] = 1
	#			adjacency_matrix[triplet[i], -1] = 1

	





	#check if the graph is fully connected
	G = nx.from_numpy_matrix(adjacency_matrix)
	is_connected = nx.is_connected(G) 
	if is_connected:
		print("Graph is connected:", is_connected)
	else:
		print("Graph is connected:", is_connected)
		components = [len(c) for c in nx.connected_components(G)]  # Get the number of nodes in each connected component
		print("Number of connected components:", len(components), "with sizes:", components)
		# get the adjacency matrix and points of the largest connected component
		#largest_component = max(nx.connected_components(G), key=len)
		#indices = np.array(list(largest_component))
		#adjacency_matrix = adjacency_matrix[indices][:, indices]
		#points = points[indices]






	return adjacency_matrix, points


import plotly.graph_objects as go
def visualize_graph_3d(adj_matrix, points_flattened, x_points=None):
	"""
	Visualizes a 3D graph using Plotly.
	
	Parameters:
		adj_matrix (numpy.ndarray): Adjacency matrix representing the graph.
		points_flattened (numpy.ndarray): Flattened array of 3D points with shape (n, 3),
										where n is the number of nodes.
	"""
	# Extract x, y, z coordinates
	x, y, z = points_flattened[:, 0], points_flattened[:, 1], points_flattened[:, 2]

	# Create edge traces
	edge_x, edge_y, edge_z = [], [], []
	for i in range(len(adj_matrix)):
		for j in range(i + 1, len(adj_matrix)):
			if adj_matrix[i, j] == 1:
				edge_x.extend([x[i], x[j], None])  # None to break the line
				edge_y.extend([y[i], y[j], None])
				edge_z.extend([z[i], z[j], None])

	# Create the graph
	fig = go.Figure()

	# Add edge traces
	fig.add_trace(go.Scatter3d(
		x=edge_x, y=edge_y, z=edge_z,
		mode='lines',
		line=dict(color='blue', width=2),
		name='Edges'
	))

	# Add node traces
	fig.add_trace(go.Scatter3d(
		x=x, y=y, z=z,
		mode='markers',
		marker=dict(size=5, color='red'),
		name='Nodes'
	))

	if x_points is not None:
		fig.add_trace(go.Scatter3d(
			x=x_points[:, 0], 
			y=x_points[:, 1], 
			z=x_points[:, 2],
			mode='markers',
			marker=dict(
				color='rgba(128, 128, 128, 0.8)',  # Grey color with 50% transparency
				size=2
			),
			name='Edges'
		))

	# Set layout
	fig.update_layout(
		scene=dict(
			xaxis_title='X Axis',
			yaxis_title='Y Axis',
			zaxis_title='Z Axis'
		),
		title="3D Graph Visualization",
		showlegend=True
	)




	fig.show()



div_time = 5000
lam3 = 0.1
path = f'master_thesis_animations/growth_wnt_alpha_bifurcation/isotropic_alpha_wnt_cells/div_time={div_time}_lam3={lam3}/sim_mes_div_time={div_time}_lam3={lam3}.npy'



x, p, q, epc = load_tree(path, scene=-1)
p = p / np.linalg.norm(p, axis=1)[:, None]

threshold = 15
number_of_points = len(x)
indices = range(number_of_points)

results, midpoints = closest_point_on_lines(x, p, threshold, indices)
print(f'Number of midpoints: {len(midpoints)}, Number of points: {len(x)}')

# Thin & sort points
thinned_points, regression_lines = thin_line(midpoints, point_cloud_thickness=2, iterations=5, sample_points=0)

sorted_points_collected = sort_points_in_segments(thinned_points, regression_lines, sorted_points_distance=2)

adjacency_matrix, points = connect_segments(sorted_points_collected, radius=15)


visualize_graph_3d(adjacency_matrix, points, x_points=x)




# Visualize the results
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# Add points to the scene
original_points = visuals.Markers(spherical=True, scaling=True)
original_points.set_data(x, face_color=[1,1,1, 1], size=1, edge_width=0)
view.add(original_points)

midpoints_marker = visuals.Markers(spherical=True, scaling=True)
midpoints_marker.set_data(midpoints, face_color='red', size=0.8, edge_width=0)
view.add(midpoints_marker)



thinned_points_marker = visuals.Markers(spherical=True, scaling=True)
thinned_points_marker.set_data(thinned_points, face_color='yellow', size=0.8, edge_width=0)
view.add(thinned_points_marker)



#index_color = np.random.rand(len(sorted_points_collected), 3)

#for idx, sorted_points in enumerate(sorted_points_collected):
#	sorted_marker = visuals.Markers(spherical=True, scaling=True)
#	sorted_marker.set_data(sorted_points, face_color=index_color[idx], size=0.7, edge_width=0)
#	view.add(sorted_marker)

#	line = scene.Line(
#		sorted_points,
#		color='blue',  # Line color
#		width=5.0,     # Line width
#		connect='strip',  # Connect points sequentially without closing the loop
#		parent=view.scene,
#	)


view.camera = 'fly'
view.camera.move_speed = 0.1  # Reduce movement speed

## Run the Vispy application
if __name__ == '__main__':
	canvas.app.run()

# make a plot of the thinned points
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(thinned_points[:, 0], thinned_points[:, 1], thinned_points[:, 2], c='y')
#ax.scatter(thinned_points[0, 0], thinned_points[0, 1], thinned_points[0, 2], c='red', s=100)
#plt.show()





