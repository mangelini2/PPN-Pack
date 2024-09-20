import torch
import trimesh
from trimesh.transformations import translation_matrix, concatenate_matrices, rotation_matrix, scale_matrix
from PIL import Image
import numpy as np
from torch.autograd import Variable
import datetime
import os
import shutil
import pdb
import time
from pyglet import gl
import matplotlib.pyplot as plt

class GeoBin():
	def __init__(self, x_max=30, y_max=30, z_max=30, thickness=2):
		self.sdf_penal = 100.0
		''' Code for making the box in Win10 with Blender api in Trimesh'''
		#box_inner = trimesh.creation.box((x_max, y_max, z_max+thickness)).apply_transform(translation_matrix([0, 0, (+thickness/2)]))
		#box_outer = trimesh.creation.box((x_max+thickness*2, y_max+thickness*2, z_max+thickness)).apply_transform(translation_matrix([0, 0, (-thickness/2)]))
		#self.box = trimesh.boolean.difference([box_outer, box_inner], engine='blender').apply_transform(translation_matrix([0, 0, -(z_max/2)]))
		#self.box.export('./box_models/box-X'+str(x_max)+'-Y'+str(y_max)+'-Z'+str(z_max)+'-T'+str(thickness)+'.obj')
		self.box = trimesh.load_mesh('./box_models/box-X'+str(x_max)+'-Y'+str(y_max)+'-Z'+str(z_max)+'-T'+str(thickness)+'.obj')
		self.scene = trimesh.scene.Scene()
		self.obj_names = ['-1']
		self.scene.add_geometry(self.box, geom_name=self.obj_names[-1])
		self.x_max = x_max
		self.y_max = y_max
		self.z_max = z_max
		self.thickness = thickness
		self.voxel_resolution = 0.2

		# Info of the objects
		self.obj_list = [] # shapely instance of each object
		self.obj_bbox_volume_list = []
		self.obj_pos = torch.tensor([]).float() # centeral location of all objects (shape [N, 3])
		self.obj_rot = torch.tensor([]).float() # rotation angle of each object (shape [N, 1]), around z-axis
		self.obj_mass_center = torch.tensor([]).float() # object mass center (shape [N, 3])
		
		self.obj_vertices = torch.tensor([]).float() # vertics of each object (shape [N, M, 3]) with 100 vertices at most
		self.vertices_mask = torch.tensor([]) # mask of the obj_vertices (shape [N, M]). 1 for existing vertices, 0 for masked-out elements
		
		self.obj_faces = torch.tensor([]).float() # vertex index of each face of each object (shape [N, F, 3]) polygon with 150 faces at most. 3 means the index of the ordered vertices.
		self.obj_face_normals = torch.tensor([]).float() # face normal of each face of each object (shape [N, F, 3]) polygon with 150 faces at most. 3 means the index of the ordered vertices.
		self.faces_mask = torch.tensor([]) # mask of the obj_faces (shape [N, F]). 1 for existing faces, 0 for masked-out elements
		
		# The global occupancy field
		self.occ = torch.zeros(x_max, y_max, z_max)
	
	def get_obj_info(self, obj):
		tmp_obj_mass_center = torch.tensor(obj.center_mass).unsqueeze(0)

		n_zero_vertices = 100
		tmp_obj_vertices = torch.cat((torch.tensor(obj.vertices).unsqueeze(0).float(), \
			torch.zeros((1, n_zero_vertices-obj.vertices.shape[0], 3))), dim=1)
		tmp_vertices_mask = torch.cat((torch.ones((1, obj.vertices.shape[0], 1)), \
			torch.zeros((1, n_zero_vertices-obj.vertices.shape[0], 1))), dim=1)
		
		n_zero_faces = 150
		tmp_obj_faces = torch.cat((torch.tensor(obj.faces).unsqueeze(0).float(), \
			torch.zeros((1, n_zero_faces-obj.faces.shape[0], 3))), dim=1).long()
		tmp_obj_face_normals = torch.cat((torch.tensor(obj.face_normals).unsqueeze(0).float(), \
			torch.zeros((1, n_zero_faces-obj.face_normals.shape[0], 3))), dim=1)
		tmp_faces_mask = torch.cat((torch.ones((1, obj.faces.shape[0], 1)), \
			torch.zeros((1, n_zero_faces-obj.faces.shape[0], 1))), dim=1)

		return tmp_obj_mass_center, tmp_obj_vertices, tmp_vertices_mask, tmp_obj_faces, tmp_obj_face_normals, tmp_faces_mask

	
	def add_object(self, obj, tmp_obj_pos=None, tmp_obj_rot=None):
		# If the object's position is not given, generate a random object location for the object
		if tmp_obj_pos is None and tmp_obj_rot is None:
			rand_x = int((np.random.rand()*self.x_max - (self.x_max/2))*0.50)
			rand_y = int((np.random.rand()*self.y_max - (self.y_max/2))*0.50)
			rand_z = int((np.random.rand()*self.z_max - self.z_max) * 0.80)
			rand_r = np.random.rand() * np.pi * 2
			tmp_obj_pos = torch.tensor([[rand_x, rand_y, rand_z]])
			tmp_obj_rot = torch.tensor([[rand_r]])
		
		# Update the object informations
		self.obj_list.append(obj)
		tmp_bounds = obj.bounds
		tmp_dims = tmp_bounds[1] - tmp_bounds[0]
		self.obj_bbox_volume_list.append(tmp_dims[0] * tmp_dims[1] * tmp_dims[2])
		self.obj_names.append(str(int(self.obj_names[-1])+1))
		self.scene.add_geometry(self.obj_list[-1], geom_name=self.obj_names[-1])
		self.obj_pos = torch.cat((self.obj_pos, tmp_obj_pos.float()), dim=0)
		self.obj_rot = torch.cat((self.obj_rot, tmp_obj_rot.float()), dim=0)
		
		tmp_obj_mass_center, tmp_obj_vertices, tmp_vertices_mask, tmp_obj_faces, tmp_obj_face_normals, tmp_faces_mask = self.get_obj_info(obj)
		self.obj_mass_center = torch.cat((self.obj_mass_center, tmp_obj_mass_center.float()), dim=0)
		self.obj_vertices = torch.cat((self.obj_vertices, tmp_obj_vertices.float()), dim=0)
		self.vertices_mask = torch.cat((self.vertices_mask, tmp_vertices_mask), dim=0)
		self.obj_faces = torch.cat((self.obj_faces, tmp_obj_faces.float()), dim=0)
		self.obj_face_normals = torch.cat((self.obj_face_normals, tmp_obj_face_normals.float()), dim=0)
		self.faces_mask = torch.cat((self.faces_mask, tmp_faces_mask), dim=0)
		return True



