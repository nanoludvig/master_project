import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.style.use(r"C:\Users\ludvi\OneDrive\Skrivebord\Nano\ludvig_mplstyle.mplstyle")

# Normalize a vector
def normalize(vector):
	magnitude = np.linalg.norm(vector)
	if magnitude == 0:
		return np.zeros_like(vector)
	return vector / magnitude

# Compute S1
def compute_S1(p_i, p_j, r_ij):
	p_i, p_j, r_ij = normalize(p_i), normalize(p_j), normalize(r_ij)
	cross_pi_ri = np.cross(p_i, r_ij)
	cross_pj_ri = np.cross(p_j, r_ij)
	return np.dot(cross_pi_ri, cross_pj_ri)

# Compute S2
def compute_S2(p_i, p_j, q_i, q_j):
	p_i, p_j, q_i, q_j = normalize(p_i), normalize(p_j), normalize(q_i), normalize(q_j)
	cross_pi_qi = np.cross(p_i, q_i)
	cross_pj_qj = np.cross(p_j, q_j)
	return np.dot(cross_pi_qi, cross_pj_qj)

# Compute S3
def compute_S3(q_i, q_j, r_ij):
	q_i, q_j, r_ij = normalize(q_i), normalize(q_j), normalize(r_ij)
	cross_qi_ri = np.cross(q_i, r_ij)
	cross_qj_ri = np.cross(q_j, r_ij)
	return np.dot(cross_qi_ri, cross_qj_ri)

# Total S calculation
def total_S(p_i, p_j, q_i, q_j, r_ij, lambdas):
	lambda_1, lambda_2, lambda_3 = lambdas
	S1 = compute_S1(p_i, p_j, r_ij)
	S2 = compute_S2(p_i, p_j, q_i, q_j)
	S3 = compute_S3(q_i, q_j, r_ij)
	return lambda_1 * S1 + lambda_2 * S2 + lambda_3 * S3

p_i = [0, 1, 0]
p_j = [0,1,0]
q_i = [1, 0, 0]
q_j = [1, 0, 0]
rij = [1, 0, 0]

print(total_S(p_i, p_j, q_i, q_j, rij, [0.5,0.4,0.1]))

# Plot particles with color based on S
def plot_colored_cells(cell1_pos, cell2_pos, polarity1, polarity2, polarity1_alt, polarity2_alt, r_ij, lambdas, cmap, filename):
	# Calculate S value
	S = total_S(polarity1, polarity2, polarity1_alt, polarity2_alt, r_ij, lambdas)
	color = cmap(1-(S + 1) / 2)  # Map S from [-1, 1] to [0, 1]
	print(filename, S)
	
	# Create plot
	fig, ax = plt.subplots(figsize=(3, 3))
	# Add a black outer frame
	frame = plt.Rectangle(
		(0, 0), 1, 1, transform=fig.transFigure,  # Cover the entire figure
		color="black", linewidth=5, fill=False, zorder=10
	)
	fig.patches.append(frame)

	cell_radius = 0.5
	ax.add_artist(plt.Circle(cell1_pos, cell_radius, facecolor=color, alpha=1, edgecolor='black', linewidth=2))
	ax.add_artist(plt.Circle(cell2_pos, cell_radius, facecolor=color, alpha=1, edgecolor='black', linewidth=2))

	# Plot polarities
	arrow_scale = 0.8
	ax.arrow(cell1_pos[0], cell1_pos[1], polarity1[0] * arrow_scale, polarity1[1] * arrow_scale, 
			width=0.05, head_width=0.2, head_length=0.2, fc='black', ec='black')
	ax.arrow(cell2_pos[0], cell2_pos[1], polarity2[0] * arrow_scale, polarity2[1] * arrow_scale, 
			width=0.05, head_width=0.2, head_length=0.2, fc='black', ec='black')
	ax.arrow(cell1_pos[0], cell1_pos[1], polarity1_alt[0] * arrow_scale, polarity1_alt[1] * arrow_scale, 
			width=0.05, head_width=0.2, head_length=0.2, fc='yellow', ec='black')
	ax.arrow(cell2_pos[0], cell2_pos[1], polarity2_alt[0] * arrow_scale, polarity2_alt[1] * arrow_scale, 
			width=0.05, head_width=0.2, head_length=0.2, fc='yellow', ec='black')
	
	# Formatting
	ax.set_xlim(-1, 2)
	ax.set_ylim(-1.5, 1.5)
	ax.set_aspect('equal')
	ax.axis('off')
	if np.abs(S) <= 1e-2:
		S=0
	ax.set_title(f"S = {S:.2f}")

	# Save the figure
	plt.tight_layout()
	plt.savefig(filename, edgecolor='black')
	#plt.show()
	plt.close()
	

lambdas = [0.5, 0.4, 0.1]  # Fixed lambda values
cmap = LinearSegmentedColormap.from_list('S_cmap', ['red', 'mediumblue'])



def configure_S_zero(lambdas):
	# Ensure S1, S2, and S3 contributions cancel out
	r_ij = (1.0, 0)  # Distance vector
	p_i = (0, 1)     # Apical-basal polarity of cell 1
	p_j = (1, 0)    # Apical-basal polarity of cell 2
	q_i = (1, 15)     # PCP polarity of cell 1
	q_j = (15, -1)    # PCP polarity of cell 2
	p_i, p_j, q_i, q_j = normalize(p_i), normalize(p_j), normalize(q_i), normalize(q_j)
	S = total_S(p_i, p_j, q_i, q_j, r_ij, lambdas)
	return p_i, p_j, q_i, q_j, r_ij, S
p_i_zero, p_j_zero, q_i_zero, q_j_zero, r_ij_zero, S_zero = configure_S_zero(lambdas)

# Existing plots (unchanged)
cases = [
	("repulsion", (0, 0), (1.0, 0), (0, 1), (0, -1), (1, 0), (1, 0), (1.0, 0)),
	("attraction", (0, 0), (1.0, 0), (0, 1), (0, 1), (1, 0), (1, 0), (1.0, 0)),
	#("intermediate", (0, 0), (1.0, 0), (0, 1), (0, -1), (1, 0), (1, 0), (1.0, 0))
]
# Add this case to the interaction plots
cases.append(
	("S_zero", (0, 0), (1.0, 0), p_i_zero, p_j_zero, q_i_zero, q_j_zero, r_ij_zero)
)

# Parameters for cases


for name, cell1_pos, cell2_pos, polarity1, polarity2, polarity1_alt, polarity2_alt, r_ij in cases:
	filename = f"{name}_interaction.png"
	polarity1, polarity1_alt, polarity2, polarity2_alt = normalize(polarity1), normalize(polarity1_alt), normalize(polarity2), normalize(polarity2_alt)
	plot_colored_cells(cell1_pos, cell2_pos, polarity1, polarity2, polarity1_alt, polarity2_alt, r_ij, lambdas, cmap, filename)



def potential(r, S, beta=5):
	return np.exp(-r) - S * np.exp(-r / beta)

# Define the radial range
r = np.linspace(0, 10, 1000)
beta = 5
# Define the range of S values and create corresponding colors
S_values = np.linspace(1, -1, 1000)
color1 = 'mediumblue'  # Color for S=1
color2 = 'red'  # Color for S=-1
cmap = LinearSegmentedColormap.from_list('gradient_cmap', [color1, color2])
# List of specific S-values from interaction plots
# Interaction S-values and positions

interaction_S_values = [
	total_S((0, 1), (0, -1), (0, 1), (0, -1), (1.0, 0), lambdas),  # antiparallel
	total_S((0, 1), (0, 1), (1, 0), (1, 0), (1.0, 0), lambdas),           # attraction
	#total_S((0, 1), (0, -1), (1, 0), (1, 0), (1.0, 0), lambdas)          # intermediate
	total_S(p_i_zero, p_j_zero, q_i_zero, q_j_zero, r_ij_zero, lambdas)  # S_zero
]

interaction_S_values = [
	{"S": -0.9, "position": (2, potential(2, -0.9, beta)), "filename": "repulsion_interaction.png"},
	{"S": 0.0, "position": (2, potential(2, 0.0, beta)), "filename": "S_zero_interaction.png"},
	{"S": 0.9, "position": (2, potential(2, 0.9, beta)), "filename": "attraction_interaction.png"}

]


# Create the figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$r_{ij}$')
ax.set_ylabel('V(r)')

# Plot each curve with its corresponding color
for i, S in enumerate(S_values):
	color = cmap(1-(i / (len(S_values) - 1)))  # Interpolate color
	ax.plot(r, potential(r, S, beta), color=color, label=f"S={S:.1f}")
ax.set_title('Potential Spectrum with Specific S-valued Interactions Highlighted', fontsize=16, fontweight='bold', pad=20)

	
for i, interaction in enumerate(interaction_S_values):
	S = interaction["S"]
	position = interaction["position"]
	filename = interaction["filename"]
	print(S)
	ax.plot(r, potential(r, S, beta), linestyle='--', color='black', linewidth=1.5, label=f"S={S:.2f}")

	placement_x = 0.34+i*0.21

	placement_y = 0.79 - i * 0.2
	# Inset for interaction figure
	inset_ax = inset_axes(ax, width="110%", height="110%", bbox_to_anchor=(placement_x, placement_y, 0.2, 0.2), 
						bbox_transform=ax.transAxes, borderpad=0)
	inset_ax.imshow(plt.imread(filename))
	inset_ax.axis('off')
	# Position on the figure to connect

	# Add a dashed line from the graph point to the figure
	ax.annotate(
		"",  # No text
		xy=(2, potential(2, S, beta)),  # Graph point
		xycoords="data",  # Data coordinates
		xytext=(placement_x, placement_y),  # Figure point
		textcoords="axes fraction",  # Fraction of figure size
		arrowprops=dict(arrowstyle="->", linestyle="-", color="black", linewidth=1),
	)

ax.set_xlim(0,10)
# Add colorbar
sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
sm.set_array([])  # Dummy data for ScalarMappable
cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
cbar.set_label('S-value')
# Disable minor ticks on the colorbar
cbar.ax.minorticks_off()
plt.savefig('potential_illustration.png',bbox_inches='tight', dpi=500)

# Display the plot
plt.show()
