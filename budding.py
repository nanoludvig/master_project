import numpy as np
import torch
from scipy.spatial.ckdtree import cKDTree
import pickle
import scipy.io
import time
import itertools
import json
import os



class Polar:
	def __init__(self, device='cpu', dtype=torch.float, init_k=100,
				callback=None):
		self.device = device
		self.dtype = dtype

		self.k = init_k
		self.true_neighbour_max = init_k//2
		self.d = None
		self.idx = None
		self.callback = callback

	@staticmethod
	def find_potential_neighbours(x, k=100, distance_upper_bound=np.inf, workers=-1):
		tree = cKDTree(x)
		d, idx = tree.query(x, k + 1, distance_upper_bound=distance_upper_bound, workers=workers)
		return d[:, 1:], idx[:, 1:]

	def find_true_neighbours(self, d, dx):
		with torch.no_grad():
			z_masks = []
			i0 = 0
			batch_size = 250
			i1 = batch_size
			while True:
				if i0 >= dx.shape[0]:
					break
				# ?
				n_dis = torch.sum((dx[i0:i1, :, None, :] / 2 - dx[i0:i1, None, :, :]) ** 2, dim=3)
				# ??
				n_dis += 1000 * torch.eye(n_dis.shape[1], device=self.device, dtype=self.dtype)[None, :, :]

				z_mask = torch.sum(n_dis < (d[i0:i1, :, None] ** 2 / 4), dim=2) <= 0
				z_masks.append(z_mask)

				if i1 > dx.shape[0]:
					break
				i0 = i1
				i1 += batch_size
		z_mask = torch.cat(z_masks, dim=0)
		return z_mask
	
	def greens_func_cc(self, idx, x):
		with torch.no_grad():
			xj = x[idx, :]
			rij = x[:, None, :] - xj
			d2 = torch.sum(rij ** 2, dim=2)

			gauss = torch.exp(-d2 / 1)
			cc = torch.sum(gauss, dim=1)
			cc_grad = torch.sum(gauss[:, :, None] * rij, dim=1)
		return cc, cc_grad



	
	def potential(self, x, p, q, idx, d, lam, alpha, cc_grad, potential):
		# Find neighbours
		full_n_list = x[idx]

		dx = x[:, None, :] - full_n_list
		z_mask = self.find_true_neighbours(d, dx)

		# Minimize size of z_mask and reorder idx and dx
		sort_idx = torch.argsort(z_mask.int(), dim=1, descending=True)

		z_mask = torch.gather(z_mask, 1, sort_idx)
		dx = torch.gather(dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))
		idx = torch.gather(idx, 1, sort_idx)

		m = torch.max(torch.sum(z_mask, dim=1)) + 1

		z_mask = z_mask[:, :m]
		dx = dx[:, :m]
		idx = idx[:, :m]

		# Normalise dx
		d = torch.sqrt(torch.sum(dx**2, dim=2))
		dx = dx / d[:, :, None]

		# Calculate S
		pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
		pj = p[idx]
		qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
		qj = q[idx]

		lam_i = lam[:, None, :].expand(p.shape[0], idx.shape[1], lam.shape[1])
		lam_j = lam[idx]
		aj = alpha[idx]
		ai = alpha[:,None,:].expand(p.shape[0], idx.shape[1],1)

		Vij = potential(x, d, dx, lam_i, lam_j, pi, pj, qi, qj, ai, aj)
		V = torch.sum(z_mask.float() * Vij)


		c_aligns_q = False
		cc_grad = None
		if cc_grad is not None:# and len(lam) >= 4 and torch.sum(torch.abs(lam[3])) > 0.0:
			if c_aligns_q:
				T1 = torch.sum(lam[:, 3][:, None] * torch.sum(torch.cross(q, cc_grad, dim=1)**2, dim=1))
				V = V + torch.sum(T1)
			else:
				T1 = torch.sum(1* p * cc_grad, dim=1)
				print(T1.sum(), V)
				V = V + torch.sum(T1)


		return V, int(m)


	def init_simulation(self, dt, lam, p, q, x, beta, alpha):
		assert len(x) == len(p)
		assert len(q) == len(x)
		assert len(beta) == len(x)

		sqrt_dt = np.sqrt(dt)
		x = torch.tensor(x, requires_grad=True, dtype=self.dtype, device=self.device)
		p = torch.tensor(p, requires_grad=True, dtype=self.dtype, device=self.device)
		q = torch.tensor(q, requires_grad=True, dtype=self.dtype, device=self.device)
		alpha = torch.tensor(alpha, requires_grad=True, dtype=self.dtype, device=self.device)

		beta = torch.tensor(beta, dtype=self.dtype, device=self.device)

		lam = torch.tensor(lam, dtype=self.dtype, device=self.device)
		# if lam is not given per-cell, return an expanded view
		if len(lam.shape) == 1:
			lam = lam.expand(x.shape[0], lam.shape[0]).clone()
		return lam, p, q, sqrt_dt, x, beta, alpha

	def update_k(self, true_neighbour_max, tstep):
		k = self.k
		fraction = true_neighbour_max / k
		if fraction < 0.25:
			k = int(0.75 * k)
		elif fraction > 0.75:
			k = int(1.5 * k)
		n_update = 1 if tstep < 50 else max([1, int(20 * np.tanh(tstep / 200))])
		self.k = k
		return k, n_update

	def time_step(self, dt, eta, lam, beta, p, q, alpha, sqrt_dt, tstep, x, r0, r1, potential, angiogenesis_fn = None, angio_params = None):
		# Start with cell division
		division, x, p, q, lam, beta, alpha, numdiv = self.cell_division(x, p, q, lam, beta, alpha, dt)


		rho=torch.sqrt(torch.sum((x-torch.tensor([-2.54168954, 3.47178755, -24.36876678], device='cuda')) ** 2, 1))
		with torch.no_grad():
			alpha[rho>r1] = 0
			alpha[rho<r1]=-0.5;
			alpha[rho<r0]=0.0;
			lam[rho<r0,0]=1;
			lam[rho<r0,1]=0;
			lam[rho<r0,2]=0;
			#lam[rho<r1,0]=0.5;
			#lam[rho<r1,1]=0.4;
			#lam[rho<r1,2]=0.1;
		alpha.requires_grad = True
		lam.requires_grad = True

		# Idea: only update _potential_ neighbours every x steps late in simulation
		# For now we do this on CPU, so transfer will be expensive
		k, n_update = self.update_k(self.true_neighbour_max, tstep)
		k = min(k, len(x) - 1)

		if division or tstep % n_update == 0 or self.idx is None:
			d, idx = self.find_potential_neighbours(x.detach().to("cpu").numpy(), k=k)
			self.idx = torch.tensor(idx, dtype=torch.long, device=self.device)
			self.d = torch.tensor(d, dtype=self.dtype, device=self.device)
		idx = self.idx
		d = self.d

		# Normalise p, q
		with torch.no_grad():
			p /= torch.sqrt(torch.sum(p ** 2, dim=1))[:, None]
			q /= torch.sqrt(torch.sum(q ** 2, dim=1))[:, None]
			q[torch.isnan(q)] = 0
		
		cc, cc_grad = self.greens_func_cc(idx, x)

		# Calculate potential
		V, self.true_neighbour_max = self.potential(x, p, q, idx, d, lam, alpha, cc_grad, potential=potential)

		# Backpropagation
		V.backward()

		# Time-step
		with torch.no_grad():
			x += -x.grad * dt + eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * sqrt_dt
			p += -p.grad * dt + eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * sqrt_dt
			#q += -q.grad * dt + eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * sqrt_dt
			if angiogenesis_fn is not None and angio_params is not None:
				x = angiogenesis_fn(x, numdiv, *angio_params)

			if self.callback is not None:
				self.callback(tstep * dt, x, p, q, lam)

		# Zero gradients
		x.grad.zero_()
		p.grad.zero_()
		q.grad.zero_()

		return x, p, q, lam, beta, alpha

	def simulation(self, x, p, q, lam, beta, alpha, eta, r0, r1, potential, yield_every=1, dt=0.1, angiogenesis_fn=None, angio_params=None):
		lam, p, q, sqrt_dt, x, beta, alpha = self.init_simulation(dt, lam, p, q, x, beta, alpha)

		tstep = 0
		while True:
			tstep += 1
			x, p, q, lam, beta, alpha = self.time_step(dt, eta, lam, beta, p, q, alpha, sqrt_dt, tstep, x, r0, r1, potential=potential, angiogenesis_fn=angiogenesis_fn, angio_params=angio_params)

			if tstep % yield_every == 0:
				xx = x.detach().to("cpu").numpy().copy()
				pp = p.detach().to("cpu").numpy().copy()
				qq = q.detach().to("cpu").numpy().copy()
				aa = alpha.detach().to("cpu").numpy().copy()
				yield xx, pp, qq, aa

	@staticmethod
	def cell_division(x, p, q, lam, beta, alpha, dt, beta_decay = 0.0):
		if torch.sum(beta) < 1e-5:
			return False, x, p, q, lam, beta, alpha, 0

		# set probability according to beta and dt
		d_prob = beta * dt
		# flip coins
		draw = torch.empty_like(beta).uniform_()
		# find successes
		events = draw < d_prob
		division = False
		numdiv = torch.sum(events)

		if numdiv > 0:
			with torch.no_grad():
				division = True
				# find cells that will divide
				idx = torch.nonzero(events)[:, 0]

				x0 = x[idx, :]
				p0 = p[idx, :]
				q0 = q[idx, :]
				l0 = lam[idx, :]
				b0 = beta[idx] * beta_decay
				a0 = alpha[idx, :]

				# make a random vector and normalize to get a random direction
				move = torch.empty_like(x0).normal_()
				move /= torch.sqrt(torch.sum(move**2, dim=1))[:, None]

				# place new cells
				x0 = x0 + move

				# append new cell data to the system state
				x = torch.cat((x, x0))
				p = torch.cat((p, p0))
				q = torch.cat((q, q0))
				lam = torch.cat((lam, l0))
				beta = torch.cat((beta, b0))
				alpha = torch.cat((alpha, a0))
				beta = torch.tensor([0.00]*len(x))
				non_zero_alpha = torch.nonzero(alpha)[0]
				beta[non_zero_alpha] = 0.1

		x.requires_grad = True
		p.requires_grad = True
		q.requires_grad = True

		return division, x, p, q, lam, beta, alpha, numdiv
	
def save(tup, name, save_dir):
	print(f'saving {save_dir}/sim_{name}.npy')
	with open(f'{save_dir}/sim_{name}.npy', 'wb') as f:
		pickle.dump(tup, f)

def angiogenesis(x, numdiv, idx, rate, angio_direction):
	x[idx,:] += rate*numdiv*angio_direction
	return x




def pot(x, d, dx, lam_i, lam_j, pi, pj, qi, qj, ai, aj):
	alphamean=(ai+aj)*0.5
	pi = pi-alphamean*dx
	pj = pj+alphamean*dx
	pi = pi/torch.sqrt(torch.sum(pi ** 2, dim=2))[:, :, None]
	pj = pj/torch.sqrt(torch.sum(pj ** 2, dim=2))[:, :, None]
	S1 = torch.sum(torch.cross(pj, dx, dim=2) * torch.cross(pi, dx, dim=2), dim=2)
	S2 = torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2)
	S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)
	S2, S3 = torch.abs(S2), torch.abs(S3)
	lam = (lam_i+lam_j)*0.5;
	S = lam[:,:,0] * S1 + lam[:,:,1] * S2 + lam[:,:,2] * S3
	Vij = (torch.exp(-d) - S * torch.exp(-d/5))
	return Vij


def initialize_pcp_around_idx(file_path, idx, r0, r1, beta):
	data = np.load(file_path, allow_pickle=True)
	x, p, q, alpha, lam = data
	x_coord = x[idx,:] + np.array([0.1, 0.1, 0])
	rho=np.sqrt(np.sum((x-x_coord) ** 2, axis=1))
	q[r1 > rho] = -np.cross(p[idx,:], (x[r1 > rho]-x_coord))
	q[r1 > rho] = q[r1 > rho] / np.linalg.norm(q[r1 > rho], axis=1, keepdims=True)
	beta = np.array([beta]*len(x))
	return x, p, q, alpha, lam, beta

def main(file_path, output_folder_path, sim_name, time_steps, x, p, q, lam, beta, alpha, eta, r0, r1, yield_every, save_every, potential, angiogenesis_fn=None, angio_params=None):
	
	try:
		os.makedirs(f'{output_folder_path}')
	except OSError:
		pass


	sim = Polar(device="cuda", init_k=50)
	runner = sim.simulation(x, p, q, lam, beta, alpha, eta=eta, r0=r0, r1=r1, yield_every=yield_every, potential=potential, angiogenesis_fn=None, angio_params=None)

	# Running the simulation
	data = []  # For storing data
	i = 0
	t0 = time.time()

	print('Starting')


	description = 'Only cells surrounding leader cell have pcp and these are all within a radius of 5. pcp is radially around leader\
		Cells between 3 and 5 away from leader cell divide and the new cells are divided towards the leader cell.\
		pcp is not updated, but given according to orientation around leader cell.\
		'
	args_dict = {
        "file_path": file_path,
        "output_folder_path": output_folder_path,
        "sim_name": sim_name,
        "time_steps": time_steps,
        "yield_every": yield_every,
        "noiselevel": eta,
        "delta_t": 0.1,
		"r0": r0,
		"r1": r1, 
		"description": description}
	
	for key, value in args_dict.items():
		if isinstance(value, np.ndarray):
			args_dict[key] = value.tolist()

	with open(__file__) as f:
		s = f.read()
	with open(f'{output_folder_path}/sim_{sim_name}.py', 'w') as f:
		f.write(s)
	with open(f'{output_folder_path}/sim_{sim_name}.json', 'w') as f:
		json.dump(args_dict, f, indent=2)



	for xx, pp, qq, aa in itertools.islice(runner, time_steps//yield_every):
		i += 1
		data.append((xx, pp, qq, aa))
		xx_data, pp_data, qq_data, aa_data = zip(*data)
		print(f'Running {i*yield_every} of {time_steps}    ({len(xx)} cells)')
		if i*yield_every % save_every == 0:
			xx_data, pp_data, qq_data, aa_data = zip(*data)
			save((xx_data, pp_data, qq_data, aa_data), sim_name, output_folder_path)
			t1 = time.time()
			total = t1-t0
			print(f'Total time: {total}')

	#if len(xx) > 3000:
	#	save((xx_data, pp_data, qq_data), 'julius_test', 'budding')
	#	print('Stopping')
	#	t1 = time.time()
	#	total = t1-t0
	#	print(f'Total time: {total}')
	#	break


file_path = 'budding/alpha_tube.npy'
time_steps = 5000
eta = 0.01
yield_every = 5
save_every = 1000
idx = 0
r0 = 6
r1 = 8
beta = 0.0

output_folder_path = 'budding/test7'
sim_name = 'test7_budding'


x, p, q, alpha, lam, beta = initialize_pcp_around_idx(file_path, idx, r0, r1, beta)


main(file_path, output_folder_path, sim_name, time_steps, x, p, q, lam, beta, alpha, eta, r0, r1, yield_every, save_every, potential=pot, angiogenesis_fn=None, angio_params=None)
