import numpy as np
import vispy.scene
from vispy.scene import visuals
import scipy
import sys

def interactive_plot(folder, alpha=10,
					only_polar=False, only_nonpolar=False):

	# Make a canvas and add simple view
	canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
	view = canvas.central_widget.add_view()
	
	
	# Load the data
	if folder.endswith('.npy'):
		data = np.load(folder, allow_pickle=True)
		x, p, q, _, _ = data
		polar_pos = x + .2 * p
		pcp_pos = x + .2 * q

	else:
		P = scipy.io.loadmat(folder);     

		x = P['x']
		p = P['p']
		q = P['q']
		

		polar_pos = x + .2 * p
		pcp_pos = x + .2 * q
	


	# Create scatter object and fill in the data
	scatter1 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
	scatter1.set_data(x, edge_width=0, face_color=(0.1, 1, 1, .5), size=2.5)

	scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
	scatter2.set_data(pcp_pos , edge_width=0, face_color='red', size=2.5)

	scatter3 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
	scatter3.set_data(polar_pos , edge_width=0, face_color='red', size=2.5)

	# Add the scatter object to the view
	if not only_polar:
		if not only_nonpolar:
			view.add(scatter1)
			view.add(scatter2)
			#view.add(scatter3)
		else:
			view.add(scatter1)
	else:
		view.add(scatter2)
		view.add(scatter3)


	# We want to fly around
	view.camera = 'fly'

	# We launch the app
	if sys.flags.interactive != 1:
		vispy.app.run()
