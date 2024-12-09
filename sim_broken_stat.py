
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import os
import re
import pandas as pd
import pickle



def list_disintegrated_files(parent_folder_path):
	"""
	Function to analyze a parent folder and return a list of full file paths where the tube has disintegrated.
	
	Parameters:
	- parent_folder_path: The parent folder path containing the simulation data in subfolders.
	
	Returns:
	- disintegrated_files: A list of full file paths where the tube has disintegrated
						(i.e., any particle has <= 3 neighbors).
	"""

	# Initialize a list to store file names where the tube has disintegrated
	disintegrated_files = []

	# Radius within which to count neighbors
	r = 3

	# Walk through the folder structure to find .npy files
	for root, dirs, files in os.walk(parent_folder_path):
		for file in files:
			if file.endswith('.npy'):
				# Construct the full file path
				full_file_path = os.path.join(root, file)

				try:
					# Load the data
					data = np.load(full_file_path, allow_pickle=True)
					x = data[0][-1]
					if len(x) == 3:
						x = data[0]
					if len(data) > 3:
						cell_type = data[3][-1]
						mes_idx = np.where(cell_type == 2)[0]
						x = np.delete(x, mes_idx, axis=0)  # Remove mesenchyme cells

					# Compute pairwise distance matrix
					dist_matrix = distance_matrix(x, x)

					# Count how many neighbors each particle has within radius r (excluding itself)
					neighbor_counts = np.sum((dist_matrix < r) & (dist_matrix > 0), axis=1)

					# Check if any particle has 3 or fewer neighbors and add to the list
					if np.any(neighbor_counts < 3):
						disintegrated_files.append(full_file_path)

				except FileNotFoundError:
					print(f"File not found: {full_file_path}")

	return disintegrated_files





def delete_and_save_last_snapshot(disintegrated_files, output_folder):
	"""
	Function to delete all snapshots except the last one from the given list of full file paths and save the updated .npy file.
	
	Parameters:
	- disintegrated_files: A list of full file paths for which the disintegration condition has been met.
	- output_folder: The folder where the updated .npy files should be saved.
	"""
	
	if len(disintegrated_files) == 0:
		print("No disintegrated files found. Exiting...")
	
	
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	
	for full_file_path in disintegrated_files:
		try:
			print(full_file_path)
			# Load the data
			data = np.load(full_file_path, allow_pickle=True)
			#print(f'data is {data}')
			
			# Extract only the last snapshot from each array in data
			new_data = []
			for array in data: #data is a tuple containing all x, p, q, cell_type...
				new_data.append(array[-1])
			tuple_of_data = tuple(new_data)

			# Construct the new file path in the output folder
			relative_path = os.path.relpath(full_file_path, start=os.path.commonpath(disintegrated_files))
			new_file_path = os.path.join(output_folder, relative_path)
			new_file_dir = os.path.dirname(new_file_path)
			if not os.path.exists(new_file_dir):
				os.makedirs(new_file_dir)
			
			with open(new_file_path, 'wb') as f:
				pickle.dump(tuple_of_data, f)

			## Save the new data to the output folder
			#np.save(new_file_path, new_data, allow_pickle=True)
			print(f"Updated file saved: {new_file_path}")
		
		except FileNotFoundError:
			print(f"File not found: {full_file_path}")
		except Exception as e:
			print(f"An error occurred while processing {full_file_path}: {e}")


def extract_parameters_from_path(file_path):
	"""
	Function to extract parameter values from a given file path.
	
	Parameters:
	- file_path: The file path containing the parameter values.
	
	Returns:
	- param_dict: A dictionary containing the extracted parameter values.
	"""
	param_dict = {}
	pattern = r'([a-zA-Z0-9_]+)=(\d*\.?\d+)'  # Regex to match parameter names and values
	matches = re.findall(pattern, file_path)
	for match in matches:
		param_name, param_value = match
		param_name = param_name.split('_')[-1]  # Keep only the part after the last underscore
		param_dict[param_name] = float(param_value)
	return param_dict

def create_disintegration_dict(parent_folder_path):
	"""
	Function to create a dictionary containing parameter values and disintegration status for each file.
	
	Parameters:
	- parent_folder_path: The parent folder path containing the simulation data in subfolders.
	
	Returns:
	- result_dict: A dictionary where keys are parameter dictionaries and values are 1 (if any particle has <= 3 neighbors) or 0 otherwise.
	"""
	result_dict = {}
	disintegrated_files = list_disintegrated_files(parent_folder_path)
	all_files = [os.path.join(root, file) for root, _, files in os.walk(parent_folder_path) for file in files if file.endswith('.npy')]
	
	for file_path in all_files:
		param_dict = extract_parameters_from_path(file_path)
		if file_path in disintegrated_files:
			result_dict[tuple(param_dict.items())] = 1
		else:
			result_dict[tuple(param_dict.items())] = 0
	print(result_dict)
	return result_dict




def plot_disintegration_grids(parent_folder_path):
	"""
	Function to create grid plots for each combination of eta and prefactor values.
	The x-axis represents lam4 values, the y-axis represents diff_cons values, and the color indicates disintegration status.
	
	Parameters:
	- parent_folder_path: The parent folder path containing the simulation data in subfolders.
	"""
	result_dict = create_disintegration_dict(parent_folder_path)

	# Extract unique values of eta and prefactor for plotting
	eta_values = set()
	prefactor_values = set()
	for params in result_dict.keys():
		param_dict = dict(params)
		if 'eta' in param_dict:
			eta_values.add(param_dict['eta'])
		if 'prefactor' in param_dict:
			prefactor_values.add(param_dict['prefactor'])
	
	# Create a grid plot for each combination of eta and prefactor
	for eta in eta_values:
		print(prefactor_values)
		print(type(prefactor_values))
		for prefactor in prefactor_values:
			lam4_values = []
			diff_cons_values = []
			statuses = []
			
			for params, status in result_dict.items():
				param_dict = dict(params)
				if param_dict.get('eta') == eta and param_dict.get('prefactor') == prefactor:
					lam4_values.append(param_dict.get('lam4', 0))
					diff_cons_values.append(param_dict.get('cons', 0))
					statuses.append(status)
			print(lam4_values, diff_cons_values, statuses)
			

			
			if lam4_values and diff_cons_values:
				# Create a 2D array (grid) to store the result for plotting
				lam4_values_sorted = sorted(list(set(lam4_values)))
				diff_cons_values_sorted = sorted(list(set(diff_cons_values)))
				result_grid = np.zeros((len(diff_cons_values_sorted), len(lam4_values_sorted)))
				
				# Populate the result grid with the values from the result_dict
				for i, diff_cons in enumerate(diff_cons_values_sorted):
					for j, lam4 in enumerate(lam4_values_sorted):
						try:
							# Use list comprehension to find the index of the tuple (diff_cons, lam4)
							idx = next(index for index, (dc, l4) in enumerate(zip(diff_cons_values, lam4_values)) 
									if (dc, l4) == (diff_cons, lam4))
							
							# If found, you can proceed to do your operation
							print(f'lam4={lam4}, diff_cons={diff_cons}')
							result_grid[i, j] = statuses[idx]
						
						except StopIteration:
							# Handle the case where the tuple is not found
							print(f'Tuple (lam4={lam4}, diff_cons={diff_cons}) not found in original values.')
				
				# Plot using seaborn heatmap
				plt.figure(figsize=(10, 8))
				ax = sns.heatmap(result_grid, annot=True, xticklabels=np.round(lam4_values_sorted, 2),
							yticklabels=np.round(diff_cons_values_sorted, 2), cmap='coolwarm', cbar=False, linewidths=0.5, linecolor='black', vmin=0, vmax=1)
				plt.xlabel('lam4')
				cbar = ax.collections[0].colorbar
				if cbar:
					cbar.set_ticks([0, 1])
					cbar.set_ticklabels(['0: Intact', '1: Disintegrated'])     
				plt.ylabel('diff_cons')
				plt.title(f'Particles with 3 or Fewer Neighbors\neta={eta}, prefactor={prefactor}')
				plt.show()




lam4=0.2
diff_cons = 1.8
string = f'lam4={lam4}_diff_cons={diff_cons}'
noise = 0.001
prefactor = 0.1



file_path = f'master_thesis_animations/single_mesenchyme/eta={noise}/prefactor={prefactor}/S4_wnt'
#delete_and_save_last_snapshot([file_path], file_path)

#print(f'the input file path is {file_path}')
plot_disintegration_grids(file_path)