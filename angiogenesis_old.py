import numpy as np
import torch
from scipy.spatial import cKDTree
import scipy.io
import os
import itertools
import gc
import scipy.io
import numpy.matlib
#import numba as nb
#from sklearn.neighbors import KDTree
import random
import pickle
import json



device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
if device == 'cuda':
	print('Using cuda')
	float_tensor = torch.cuda.FloatTensor
else:
	print('Using cpu')
	float_tensor = torch.FloatTensor

def loadsphere(file_path, l1_0, l2_0, pdiv):
	if type(file_path) == str:
		np.random.seed(103833)
		_, ext = os.path.splitext(file_path)
		if ext == '.mat':
			# Load .mat file
			data = scipy.io.loadmat(file_path)
			x = data['x']
			p = data['p']
			q = data['q']
		elif ext == '.txt':
			# Load .txt file
			with open(file_path, 'r') as f:
				data = eval(f.read())
			x = data['x']
			p = data['p']
			q = data['q']
		else:
			raise ValueError('Unsupported file type. Please use .mat or .txt file.')	
	else:
		# Load dictionary
		x = file_path['x']
		p = file_path['p']
		q = file_path['q']

	# Center the structure:
	x = x-np.mean(x)

	lam = np.array([l1_0, l2_0, 1-l1_0-l2_0])
	q[:,:] = 0
	angio_direction = p[0,:]
	mask2 = np.where(np.linalg.norm((x-x[0,:]),axis=1)<7)
	q[mask2] = -np.cross(x[mask2]-x[0,:], angio_direction)
	mask3 = np.where(np.linalg.norm((x-x[0,:]),axis=1)<2)
	q[mask3] = 0
	return x, p, q, lam, pdiv

'''
def find_potential_neighbours(x, k=100, distance_upper_bound=np.inf):
	tree = cKDTree(x)
	d, idx = tree.query(x, k + 1, distance_upper_bound=distance_upper_bound, workers=-1)
	return d[:, 1:], idx[:, 1:]
'''
def find_potential_neighbours(x, k=100, distance_upper_bound=np.inf, domain_size=None,
                              cyclic=True):
    assert x.shape[0] > 1
    k = min(k + 1, x.shape[0])

    if cyclic:
        tree = cKDTree(x, boxsize=domain_size, balanced_tree=True, compact_nodes=True)
    else:
        tree = cKDTree(x, balanced_tree=True, compact_nodes=True)

    d, idx = tree.query(x, k, distance_upper_bound=distance_upper_bound)

    return d[:, 1:], idx[:, 1:]


def find_true_neighbours(d, dx):
	with torch.no_grad():
		z_masks = []
		i0 = 0
		batch_size = 500
		i1 = batch_size
		while True:
			if i0 >= dx.shape[0]:
				break

			n_dis = torch.sum((dx[i0:i1, :, None, :] / 2 - dx[i0:i1, None, :, :]) ** 2, dim=3)
			n_dis += 1000 * torch.eye(n_dis.shape[1], device=device)[None, :, :]

			z_mask = torch.sum(n_dis < (d[i0:i1, :, None] ** 2 / 4), dim=2) <= 0  # check summatio dimension, etc.
			z_masks.append(z_mask)

			if i1 > dx.shape[0]:
				break
			i0 = i1
			i1 += batch_size
	z_mask = torch.cat(z_masks, dim=0)
	return z_mask


def divide_into_center(x, p, q, pdiv, device='cuda'):
	#center = torch.tensor([ -1.4155,   2.8733, -28.1903], device=device)
	center = x[0,:]
	distances = torch.linalg.vector_norm(x - center, dim=1)
	mask = torch.where((6 < distances) & (distances < 7))
	# now probabilistically choose which of these cells to divide based on pdiv:
	dividing = torch.rand(mask[0].shape[0], device=device) < pdiv
	dividing_indices = mask[0][dividing]
	numdiv = dividing_indices.shape[0]
	angio_direction = torch.tensor([-0.9300,  0.3571, -0.0889], device=device)

	if numdiv > 0:
		with torch.no_grad():
			# The new cells should divide towards the center
			new_x = x[dividing_indices] + (center-x[dividing_indices])/torch.linalg.vector_norm(x[dividing_indices] - center)
			new_p = p[dividing_indices]
			new_q = torch.cross(angio_direction.unsqueeze(0), new_x-center)
		x = torch.cat((x.detach(), new_x), dim=0)
		p = torch.cat((p.detach(), new_p), dim=0)
		q = torch.cat((q.detach(), new_q), dim=0)
		x.requires_grad = True
		p.requires_grad = True
		q.requires_grad = True
		assert q.shape == x.shape
		assert x.shape == p.shape
	return x, p, q, numdiv




def init_simulation(dt, lam, p, x, q, pdiv):
	sqrt_dt = np.sqrt(dt)
	x = torch.tensor(x, requires_grad=True, dtype=torch.float, device=device)
	p = torch.tensor(p, requires_grad=True, dtype=torch.float, device=device)
	q = torch.tensor(q, requires_grad=True, dtype=torch.float, device=device)
	lam = torch.tensor(lam, dtype=torch.float, device=device)
	pdiv = torch.tensor(pdiv, dtype=torch.float, device=device)
	return lam, p, sqrt_dt, x, q, pdiv



def potential(x, p, q, idx, d, lam, z_mask, dx,m, polarity):
	
	if polarity == True:
		# Calculate S
		pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
		pj = p[idx]
		qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
		qj = q[idx]
		pi = pi/torch.sqrt(torch.sum(pi ** 2, dim=2))[:, :, None]
		pj = pj/torch.sqrt(torch.sum(pj ** 2, dim=2))[:, :, None]
		S1 = torch.sum(torch.cross(pj, dx, dim=2) * torch.cross(pi, dx, dim=2), dim=2)
		S2 = torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2)
		S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)
		#S1, S2 = torch.abs(S1), torch.abs(S2)
		lambda1 = torch.full((S1.shape[0], 1), lam[0], device=device)
		mask_tensor = torch.where((q == 0).all(dim=1))[0].to(device)
		lambda1[mask_tensor] = 1
		S = lambda1*S1+lam[1]*S2+lam[2]*S3 

		# Potential
		Vij = z_mask.float() * (torch.exp(-d) - S * torch.exp(-d/5))
		V = torch.sum(Vij)
	if polarity == False:
		Vij = z_mask.float() * (torch.exp(-d) - 1*torch.exp(-d/5))
		V = torch.sum(Vij)

	return V, int(m)

def save(tup, name, save_dir):
	print(f'saving {save_dir}/sim_{name}.npy')
	with open(f'{save_dir}/sim_{name}.npy', 'wb') as f:
		pickle.dump(tup, f)


class TimeStepper:
	
	def __init__(self, init_k):
		self.k = init_k
		self.true_neighbour_max = init_k//2
		self.d = None
		self.idx = None
		self.noise = float_tensor(1)

	def update_k(self, true_neighbour_max, tstep):
		k = self.k
		fraction = true_neighbour_max / k
		if fraction < 0.25:
			k = int(0.75 * k)
		elif fraction > 0.75:
			k = int(1.5 * k)
		n_update = 1 if tstep < 150 else max([1, int(20 * np.tanh(tstep / 200))])
		self.k = k
		return k, n_update
	

	def update_noise(self, x):
		if self.noise.shape != x.shape:
			self.noise = float_tensor(*x.shape)

		self.noise.normal_()
		return self.noise
	
	def time_step(self, dt, eta, lam, p, sqrt_dt, tstep, x, q, pdiv, idx, polarity, dynamic):

		angio_direction = torch.tensor([-0.9300,  0.3571, -0.0889], device=device)


		x, p, q, numdiv = divide_into_center(x, p, q, pdiv)
		
		k, n_update = self.update_k(self.true_neighbour_max, tstep)
		if tstep % n_update == 0 or self.idx is None or numdiv>0:
			d, idx = find_potential_neighbours(x.detach().to("cpu").numpy(), k=k)
			self.idx = torch.tensor(idx, dtype=torch.long, device=device)
			self.d = torch.tensor(d, dtype=torch.float, device=device)
		idx = self.idx
		d = self.d

		# Normalise p, q
		with torch.no_grad():
			p /= torch.sqrt(torch.sum(p ** 2, dim=1))[:, None]
			q /= torch.sqrt(torch.sum(q ** 2, dim=1))[:, None]
			q[torch.isnan(q)] = 0

		
		# Find true neighbours
		full_n_list = x[idx]
		dx = x[:, None, :] - full_n_list
		z_mask = find_true_neighbours(d, dx)
		# Minimize size of z_mask and reorder idx and dx
		#sort_idx = torch.argsort(z_mask, dim=1, descending=True)
		_, sort_idx = torch.topk(z_mask.to(torch.int), k=z_mask.size(1), dim=1)
		z_mask = torch.gather(z_mask, 1, sort_idx)
		dx = torch.gather(dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))
		idx = torch.gather(idx, 1, sort_idx)
		m = torch.max(torch.sum(z_mask, dim=1)) + 1
		z_mask = z_mask[:, :m]
		dx = dx[:, :m]
		idx = idx[:, :m]
		# Normalize dx
		d = torch.sqrt(torch.sum(dx**2, dim=2))
		dx = dx / d[:, :, None]
	

		# Calculate potential
		V, self.true_neighbour_max = potential(x, p, q, idx, d, lam, z_mask, dx, m, polarity)

		# Backpropagation
		V.backward()
		mask = torch.where((q == 0).all(dim=1))[0].to(device)
		# Time-step
		with torch.no_grad():
			x += -x.grad * dt + eta * self.update_noise(x) * sqrt_dt
			x[0,:] += 0.1*numdiv*angio_direction
		if polarity == True and dynamic==True:
			with torch.no_grad():
				p += -p.grad * dt + eta * self.update_noise(x) * sqrt_dt
				#q += -q.grad * dt + eta * self.update_noise(x) * sqrt_dt
				q[mask] = 0
				p.grad.zero_()
				q.grad.zero_()

				



		# Zero gradients
		x.grad.zero_()
		numdiv=0

		return x, p, q, lam, V, pdiv, idx
	
def simulation(x, p, q, lam, pdiv,  eta, polarity, dynamic, dt, yield_every=1):
	lam, p, sqrt_dt, x, q, pdiv, = init_simulation(dt, lam, p, x, q, pdiv) 
	time_stepper = TimeStepper(init_k=200)
	tstep = 0
	idx= torch.tensor(np.zeros((1000,3)), dtype=torch.int, device=device)
	while True:
		tstep +=1
		x, p, q, lam, V, pdiv, idx = time_stepper.time_step(dt, eta, lam, p, sqrt_dt, tstep, x, q, pdiv,  idx, polarity, dynamic)


		if tstep % yield_every == 0:
			xx = x.detach().to("cpu").numpy()
			pp = p.detach().to("cpu").numpy()
			qq = q.detach().to("cpu").numpy()
			VV = V.detach().to("cpu").numpy()
			yield xx, pp, qq, VV

		gc.collect()


def main(file_path, output_folder_path, sim_name, n_steps=5000, fraction_of_frames=10, noiselevel=0.05, delta_t=0.1, l1_0=0.6, l2_0=0.3, pdiv=0.0001, polarity=False, dynamic=False):
	x, p, q, lam, pdiv = loadsphere(file_path, l1_0, l2_0, pdiv)


	try:
		os.makedirs(f'{output_folder_path}')
	except OSError:
		pass

	x_data = []
	p_data = []
	q_data = []
	V_data = []
	description = 'Only cells surrounding leader cell have pcp and these are all within a radius of 5. pcp is radially around leader\
		Cells between 3 and 5 away from leader cell divide and the new cells are divided towards the leader cell.\
		pcp is not updated, but given according to orientation around leader cell.\
		'
	args_dict = {
        "file_path": file_path,
        "output_folder_path": output_folder_path,
        "sim_name": sim_name,
        "n_steps": n_steps,
        "fraction_of_frames": fraction_of_frames,
        "noiselevel": noiselevel,
        "delta_t": delta_t,
        "l1_0": l1_0,
        "l2_0": l2_0,
        "pdiv": pdiv,
        "polarity": polarity,
        "dynamic": dynamic,
		"description": description}
	
	for key, value in args_dict.items():
		if isinstance(value, np.ndarray):
			args_dict[key] = value.tolist()


	xx=x
	pp=p
	qq=q
	i = 0
	N = n_steps
	save((xx, pp, qq, V_data), sim_name, output_folder_path)
	with open(__file__) as f:
		s = f.read()
	with open(f'{output_folder_path}/sim_{sim_name}.py', 'w') as f:
		f.write(s)
	with open(f'{output_folder_path}/sim_{sim_name}.json', 'w') as f:
		json.dump(args_dict, f, indent=2)
	for xx, pp, qq, VV in itertools.islice(simulation(x, p, q, lam, pdiv, eta=noiselevel, yield_every=fraction_of_frames, dt=delta_t, polarity=polarity, dynamic=dynamic), int(N//fraction_of_frames)):
		i += 1
		x_data.append(xx)
		p_data.append(pp)
		q_data.append(qq)
		V_data.append(VV)
		print(f'Running {i} of {N//fraction_of_frames}, {i*fraction_of_frames} steps of {N} in total, particles = {xx.shape[0]}, name = {sim_name}', end='\n')
		if i > 5 and i % 10 == 0:
			save((x_data, p_data, q_data, V_data), sim_name, output_folder_path)
			gc.collect()
	print(f'Simulation done, saved {len(x_data)} datapoints')

	save((x_data, p_data, q_data, V_data), sim_name, output_folder_path)





#if __name__ == '__main__':	
#	main(f'test/tubify/tubify_sphere1000_pcp_cut/t100000.mat', f'budding/test1', 'test1_budding', 10000, 20, 0.00, 0.2, 0.5, 0.42, 0.005, True, True)


if __name__ == '__main__':	
	main(f'budding/alpha_test_file.npy', f'budding/test6', 'test6_budding', 10000, 2, 0.00, 0.2, 0.5, 0.42, 0.005, True, True)


