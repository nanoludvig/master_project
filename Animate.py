import numpy as np
import vispy.scene
from vispy.scene import visuals
import scipy.io
import sys
from vispy import app, scene
import imageio
from glob import glob
import re


iterator = 0
def interactive_animate(folder, alpha=10, view_particles=None, interval=1/60):
	
	# Getting the data
	x_lst = []
	polar_pos_lst = []
	files = glob(folder + '/t*.mat' )
	files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[-1]))

	# Check if the folder contains MATLAB files
	if files:
		for filename in files:
			P = scipy.io.loadmat(filename)                    
			x = P['x']
			p = P['p']
			q = P['q']
			polar_pos = x + .2 * q
			x_lst.append(x)
			polar_pos_lst.append(polar_pos)
	else:
		# If no MATLAB files, try to read the .npy file
		npy_file = folder
		data = np.load(npy_file, allow_pickle=True)

		if len(data) == 4:

			x_data, p_data, q_data, a_data = data

			for i in range(len(x_data)):
				x = x_data[i]
				p = p_data[i]
				q = q_data[i]
				a = a_data[i]
				polar_pos = x + .2 * q
				x_lst.append(x)
				polar_pos_lst.append(polar_pos)
				#alpha_list.append(x*np.abs(a)/np.max(np.abs(a)))
			files = x_data
			print(f'len(files): {len(files)}')
		elif len(data) == 3:
			x_data, p_data, q_data = data

			for i in range(len(x_data)):
				x = x_data[i]
				p = p_data[i]
				q = q_data[i]
				polar_pos = x + .2 * q
				x_lst.append(x)
				polar_pos_lst.append(polar_pos)
			files = x_data
			print(f'len(files): {len(files)}')
		else:
			x_data, p_data, q_data, a_data, l_data = data
			alpha_list = []

			for i in range(len(x_data)):
				x = x_data[i]
				p = p_data[i]
				q = q_data[i]
				a = a_data[i]
				l = l_data[i]
				polar_pos = x + .2 * q * l[:,2].reshape(-1,1)/l[:,2].reshape(-1,1).max()
				x_lst.append(x)
				polar_pos_lst.append(polar_pos)
				alpha_list.append(x*np.abs(a)/np.max(np.abs(a)))
			files = x_data
			print(f'len(files): {len(files)}')


	# Make a canvas and add simple view
	canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
	view = canvas.central_widget.add_view()                                   

	# Create scatter object and fill in the data
	scatter1 = visuals.Markers(scaling=True, alpha=10, spherical=True)
	scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
	scatter3 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
	if len(data) == 5:
		special_scatter = visuals.Markers(scaling=True, alpha=1, spherical=True)

	scatter1.set_data(x_lst[0] , edge_width=0, face_color=(0.1, 1, 1, .5), size=2.5)
	scatter2.set_data(x_lst[0] , edge_width=0, face_color=(1, 1, 1, .5), size=2.5)
	scatter3.set_data(polar_pos_lst[0] , edge_width=0, face_color='red', size=2.5)
	if len(data) == 5:
		special_scatter.set_data(alpha_list[0], edge_width=0, face_color='yellow', size=3)  # Choose a standout color and size


	# Add the scatter object to the view
	if not view_particles:
		view.add(scatter1)
		view.add(scatter2)
		view.add(scatter3)
		if len(data) == 5:
			view.add(special_scatter)
	else:
		assert view_particles == "polar" or view_particles =="non_polar", "view_particles only takes arguments polar or non_polar"
		if view_particles == 'polar':
			view.add(scatter2)
			view.add(scatter3)
			if len(data) == 5:
				view.add(special_scatter)
		if view_particles == 'non_polar':
			view.add(scatter1)


	def update(ev):
		global x, iterator
		iterator += 1
		x = x_lst[int(iterator) % len(files)]
		polar_pos = polar_pos_lst[int(iterator) % len(files)]
		if len(data) == 5:
			special_points = alpha_list[int(iterator) % len(files)]
		scatter1.set_data(x, edge_width=0, face_color=(0.1, 1, 1, .5), size=2.5)
		scatter2.set_data(x, edge_width=0, face_color=(1, 1, 1, .5), size=2.5)
		scatter3.set_data(polar_pos , edge_width=0, face_color='red', size=2.5)
		if len(data) == 5:
			special_scatter.set_data(special_points, edge_width=0, face_color='yellow', size=3, symbol='diamond')


	timer = app.Timer(interval=interval)
	timer.connect(update)
	timer.start()

	#@canvas.connect
	#def on_key_press(event):
	#    if event.text == ' ':
	#        if timer.running:
	#            timer.stop()
	#        else:
	#            timer.start()
	@canvas.connect
	def on_key_press(event):
		global iterator
		if event.text == ' ':
			if timer.running:
				timer.stop()
			else:
				timer.start()
		elif event.text == 'r':
			iterator -= 51
			update(1)
		elif event.text == 't':
			iterator += 49
			update(1)
		elif event.text == ',':
			iterator -= 2
			update(1)
		elif event.text == '.':
			update(1)

	# We want to fly around
	view.camera = 'arcball'
	


	# We launch the app
	if sys.flags.interactive != 1:
		vispy.app.run()
		


def export_gif(folder, output_name, alpha=10,
			view_particles=None):

	# Getting the data
	x_lst = []
	polar_pos_lst = []
	mask = np.load(folder + '/p_mask.npy')
	files = glob(folder + '/t*.mat' )
	files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[-1]))
	for filename in files:
		P = scipy.io.loadmat(filename)                    
		x = P['x']
		p = P['p']
		polar_pos = x + .2 * p
		x_lst.append(x)
		polar_pos_lst.append(polar_pos)

	x_lst = np.array(x_lst)
	polar_pos_lst = np.array(polar_pos_lst)

	###

	# Make a canvas and add simple view
	canvas = vispy.scene.SceneCanvas(keys='interactive', bgcolor='black',
							size=(1200, 800), show=True)
	view = canvas.central_widget.add_view()                                   

	# Create scatter object and fill in the data
	scatter1 = visuals.Markers(scaling=True, alpha=10, spherical=True)
	scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
	scatter3 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)

	scatter1.set_data(x_lst[0][mask==0] , edge_width=0, face_color=(0.1, 1, 1, .5), size=2.5)
	scatter2.set_data(x_lst[0][mask==1] , edge_width=0, face_color=(1, 1, 1, .5), size=2.5)
	scatter3.set_data(polar_pos_lst[0] , edge_width=0, face_color='red', size=2.5)

	view = canvas.central_widget.add_view()
	view.camera = 'arcball'
	view.camera.distance = 70

	# Create scatter object and fill in the data
	scatter1 = visuals.Markers(scaling=True, alpha=10, spherical=True)
	scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
	scatter3 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)

	scatter1.set_data(x_lst[0][mask==0] , edge_width=0, face_color=(0.1, 1, 1, .5), size=2.5)
	scatter2.set_data(x_lst[0][mask==1] , edge_width=0, face_color=(1, 1, 1, .5), size=2.5)
	scatter3.set_data(polar_pos_lst[0] , edge_width=0, face_color='red', size=2.5)

	# Add the scatter object to the view
	if not view_particles:
		view.add(scatter1)
		view.add(scatter2)
		view.add(scatter3)
	else:
		assert view_particles == "polar" or view_particles =="non_polar", "view_particles only takes arguments polar or non_polar"
		if view_particles == 'polar':
			view.add(scatter2)
			view.add(scatter3)
		if view_particles == 'non_polar':
			view.add(scatter1)

	output_filename = f'{output_name}.mp4'

	writer = imageio.get_writer(output_filename)

	iterator = 0
	for i in range(len(x_lst)):
		im = canvas.render()
		writer.append_data(im)
		iterator += 1
		x = x_lst[int(iterator) % len(files)]
		polar_pos = polar_pos_lst[int(iterator) % len(files)]
		scatter1.set_data(x[mask==0] , edge_width=0, face_color=(0.1, 1, 1, .5), size=2.5)
		scatter2.set_data(x[mask==1] , edge_width=0, face_color=(1, 1, 1, .5), size=2.5)
		scatter3.set_data(polar_pos , edge_width=0, face_color='red', size=2.5)
		view.camera.transform.rotate(1, axis=[0,0,1])
	writer.close()

	"""
	
		
	# Getting the data
	x_lst = []
	polar_pos_lst = []
	mask = np.load(folder + '/p_mask.npy')
	files = glob(folder + '/t*.mat' )
	files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[-1]))
	for filename in files:
		P = scipy.io.loadmat(filename)                    
		x = P['x']
		p = P['p']
		polar_pos = x[mask == 1] + .2 * p[mask == 1]
		x_lst.append(x)
		polar_pos_lst.append(polar_pos)

	x_lst = np.array(x_lst)
	polar_pos_lst = np.array(polar_pos_lst)

	# Make a canvas and add simple view
	canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
	view = canvas.central_widget.add_view()                                   

	# Create scatter object and fill in the data
	scatter1 = visuals.Markers(scaling=True, alpha=10, spherical=True)
	scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
	scatter3 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)

	scatter1.set_data(x_lst[0][mask==0] , edge_width=0, face_color=(0.1, 1, 1, .5), size=2.5)
	scatter2.set_data(x_lst[0][mask==1] , edge_width=0, face_color=(1, 1, 1, .5), size=2.5)
	scatter3.set_data(polar_pos_lst[0] , edge_width=0, face_color='red', size=2.5)

	# Add the scatter object to the view
	if not view_particles:
		view.add(scatter1)
		view.add(scatter2)
		view.add(scatter3)
	else:
		assert view_particles == "polar" or view_particles =="non_polar", "view_particles only takes arguments polar or non_polar"
		if view_particles == 'polar':
			view.add(scatter2)
			view.add(scatter3)
		if view_particles == 'non_polar':
			view.add(scatter1)


	def update(ev):
		global x, iterator
		iterator += 1
		x = x_lst[int(iterator) % len(files)]
		polar_pos = polar_pos_lst[int(iterator) % len(files)]
		scatter1.set_data(x[mask==0] , edge_width=0, face_color=(0.1, 1, 1, .5), size=2.5)
		scatter2.set_data(x[mask==1] , edge_width=0, face_color=(1, 1, 1, .5), size=2.5)
		scatter3.set_data(polar_pos , edge_width=0, face_color='red', size=2.5)


	timer = app.Timer(interval=interval)
	timer.connect(update)
	timer.start()

	@canvas.connect
	def on_key_press(event):
		if event.text == ' ':
			if timer.running:
				timer.stop()
			else:
				timer.start()

	# We want to fly around
	view.camera = 'fly'

	# We launch the app
	if sys.flags.interactive != 1:
		vispy.app.run()

	"""