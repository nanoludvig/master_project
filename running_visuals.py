import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import sys
from sklearn.decomposition import PCA
#from mayavi import mlab
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation



lam4=0.3
prefactor = 0.02
alpha = 0.5

folder = f'master_thesis_animations/growth_wnt_alpha_bifurcation/isotropic_alpha_wnt_cells/alpha={alpha}_lam4={lam4}_prefactor={prefactor}/sim_alpha={alpha}_lam4={lam4}_prefactor={prefactor}.npy'
data = np.load(folder, allow_pickle=True)
x, p_original, q_original = data[0], data[1], data[2]


displacement_list = []
final_shape = x[-1].shape[0]  # Total number of particles in the final frame

# Pad the initial positions array to match the largest array
x_0_padded = np.pad(x[0], ((0, final_shape - x[0].shape[0]), (0, 0)), mode='constant', constant_values=0)
frame_padded_list = []
# Loop through each frame to calculate displacement
for frame in x:
    # Pad the current frame to match the final shape
    frame_padded = np.pad(frame, ((0, final_shape - frame.shape[0]), (0, 0)), mode='constant', constant_values=0)
    frame_padded_list.append(frame_padded)
    
    # Calculate the displacement relative to the initial padded positions
    displacement = frame_padded - x_0_padded
    displacement_list.append(displacement)

displacement_array = np.array(displacement_list)  # Shape: (timesteps, num_particles, 3)
frame_array = np.array(frame_padded_list)  # Shape: (timesteps, num_particles, 3)'

# Initialize an array to store total distances for each particle
num_particles = frame_array.shape[1]
total_distance_traveled = np.zeros(num_particles)

# Loop through each frame (start from the second frame to calculate differences)
for i in range(1, frame_array.shape[0]):
    # Get the positions in the current and previous frames
    current_positions = frame_array[i]
    previous_positions = frame_array[i - 1]
    
    # Calculate a mask to check where particles are non-zero in both frames
    valid_mask = (np.linalg.norm(current_positions, axis=1) > 0) & (np.linalg.norm(previous_positions, axis=1) > 0)
    
    # Calculate the displacement only for particles that exist in both frames
    displacement = current_positions[valid_mask] - previous_positions[valid_mask]
    
    # Calculate the Euclidean distance for each valid particle
    distance = np.linalg.norm(displacement, axis=1)
    
    # Accumulate distances for particles that are valid in both frames
    total_distance_traveled[valid_mask] += distance

# Sort particles by total distance traveled in descending order
sorted_indices = np.argsort(-total_distance_traveled)
sorted_distances = total_distance_traveled[sorted_indices]

# Display the top 5 particles with the highest distance traveled
print("Top 5 particles with the highest distance traveled:")
for i in range(5):
    print(f"Particle {sorted_indices[i]}: Total distance = {sorted_distances[i]}")
    
## Plot the trajectories
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.set_title("3D Trajectories of Particles")

## Plot each particle's trajectory
#particle_idx = 1000
#trajectory = frame_array[:, particle_idx, :]

## Filter out rows where the trajectory has only zeros (for particles not present in all timesteps)
#non_zero_trajectory = trajectory[~np.all(trajectory == 0, axis=1)]

## Plot the trajectory if the particle has valid data
#if non_zero_trajectory.size > 0:
#	ax.plot(non_zero_trajectory[:, 0], non_zero_trajectory[:, 1], non_zero_trajectory[:, 2])
#final_positions = frame_array[-1]
#ax.scatter(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], 
#           color='red', s=50, label='Final Structure', alpha=0.1)
## Set labels and show plot
#ax.set_xlabel("X")
#ax.set_ylabel("Y")
#ax.set_zlabel("Z")
#plt.show()






'''
# Convert x to spherical coordinates
def cartesian_to_spherical(x):
    x, y, z = x[:, 0], x[:, 1], x[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # polar angle
    phi = np.arctan2(y, x)    # azimuthal angle
    return r, theta, phi

# Convert spherical to Cartesian coordinates
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack((x, y, z), axis=-1)



np.set_printoptions(threshold=sys.maxsize)

folder = 'master_thesis_animations/curl_around_bud.npy'
data = np.load(folder, allow_pickle=True)
x_original, p_original, q_original = data[0], data[1], data[2]

# Create a mask that includes only the points with 400 indices and where x[:,0] > 3
mask = (np.arange(len(x_original)) > 0) & (x_original[:, 0] > 3)

# Apply the mask to x and q
x = x_original[mask]
p = p_original[mask]
q = q_original[mask]


x = x - np.mean(x, axis=0)
# Define the vector field with a -1/2 topological defect
def vector_field(theta, defect_number):
    # Direction of the vector field in terms of theta
    vx = np.cos(theta*defect_number)
    vy = np.sin(theta*defect_number)
    return vx, vy
'''
'''
X, Y = x[:,1], x[:,2]
R, Theta = np.hypot(X, Y), np.arctan2(Y, X)
defect_number_fraction = Fraction(1, 2)
defect_number = float(defect_number_fraction)
vy, vz = vector_field(Theta, defect_number)
vx = np.zeros_like(vy)




# Use NearestNeighbors to find neighbors for normal approximation
nbrs = NearestNeighbors(n_neighbors=4, algorithm='auto').fit(x)  # Using 4 neighbors to define a plane
distances, indices = nbrs.kneighbors(x)

# Choose a point to visualize the plane defined by the normal vector
point_idx = 286  # Index of the point to visualize
point = x[point_idx]
neighbors = x[indices[point_idx][1:]]  # Exclude the point itself
# Calculate two vectors in the plane using neighboring points
vec1 = neighbors[1] - neighbors[0]
vec2 = neighbors[2] - neighbors[0]

# Calculate the normal vector to the plane
normal = np.cross(vec1, vec2)
normal /= np.linalg.norm(normal)

# Define the plane
plane_size = 2.0
u = np.linspace(-plane_size, plane_size, 10)
v = np.linspace(-plane_size, plane_size, 10)
u, v = np.meshgrid(u, v)
plane_x = point[0] + u
plane_y = point[1] + v
plane_z = point[2] + (-normal[0] * u - normal[1] * v) / normal[2]

# Plot the point, normal vector, and plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], length=1.0, color='r', label='Normal Vector')
ax.plot_surface(plane_x, plane_y, plane_z, alpha=0.5, color='b')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], color='k', s=10, label='Points')
ax.scatter(point[0], point[1], point[2], color='g', s=50, label='Selected Point')

initial_position = point + np.array([3, 0, 0])  # Start 3 units above in the x-direction
num_frames = 50
rotation_frames = 50

# Initialize the vector to animate
vector_plot = None

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# plt.legend(loc='upper left')

# Animation function
def update(num):
    global vector_plot
    if vector_plot:
        vector_plot.remove()  # Remove the previous vector
    if num < num_frames:
        # Falling down phase
        t = num / num_frames
        current_position = initial_position * (1 - t) + point * t
        vector_plot = ax.quiver(current_position[0], current_position[1], current_position[2], vx[point_idx], vy[point_idx], vz[point_idx], color='m', length=1.0)
    else:
        # Rotating phase
        rotation_t = (num - num_frames) / rotation_frames
        angle = np.pi / 2 * rotation_t  # Rotate up to 90 degrees
        # Define rotation axis
        rotation_axis = np.array([0, 0, 1])
        # Use Rodrigues' rotation formula to rotate vector
        v = np.array([vx[point_idx], vy[point_idx], vz[point_idx]])
        k = rotation_axis
        v_rot = (v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * np.dot(k, v) * (1 - np.cos(angle)))
        v_rot /= np.linalg.norm(v_rot)
        vector_plot = ax.quiver(point[0], point[1], point[2], v_rot[0], v_rot[1], v_rot[2], color='m', length=1.0)
        # Stop when vector is in the plane (approximately normal to the normal vector)
        if np.abs(np.dot(v_rot, normal)) < 1e-2:
            ani.event_source.stop()

ani = animation.FuncAnimation(fig, update, frames=num_frames + rotation_frames, interval=50, repeat=False)
plt.show()
'''
'''
# make matplorlib quiver plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(x[:,0], x[:,1], x[:,2], q[:,0], q[:,1], q[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#plt.show()

#q_original[mask] = q

output_folder = 'master_thesis_animations/plus_one_half_defect_on_bud.npy'
#np.save(output_folder, [x_original, p_original, q_original])

Y, Z = x[:,1], x[:,2]
R, Theta = np.hypot(Y, Z), np.arctan2(Z, Y)
defect_number_fraction = Fraction(1, 2)
defect_number = float(defect_number_fraction)
q[:,1], q[:,2] = vector_field(Theta, defect_number)

# make matplorlib quiver plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(x[:,0], x[:,1], x[:,2], q[:,0], q[:,1], q[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


# Plot the vector field
plt.figure(figsize=(8, 8))
plt.quiver(Y, Z, q[:,1], q[:,2], color='blue', scale=20, headwidth=3)
plt.xlabel('Y')
plt.ylabel('Z')
plt.title(f'Vector Field With a {defect_number_fraction} Defect')
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
'''