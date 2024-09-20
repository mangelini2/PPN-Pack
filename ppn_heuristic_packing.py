import torch
import trimesh
from trimesh.transformations import translation_matrix, concatenate_matrices, rotation_matrix, scale_matrix, decompose_matrix
from PIL import Image, ImageFilter
import numpy as np
from torch.autograd import Variable
import datetime
import os
import shutil
import pdb
import time
import glob
import pdb
import copy
import sys
import cv2

import torch
import torch.nn.functional as F
import pybullet as p

from ppnpack_models import *

def compute_sdf_kernal(use_cuda=False):
	x = torch.arange(10+1)
	y = torch.arange(10+1)
	z = torch.arange(10*5+1)
	z_scaling = torch.tensor([1.0,1.0,0.2])
	center = torch.tensor([[5.0,5.0,25.0]])
	grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
	grid_pts = torch.cat((grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1,1)), dim=-1).unsqueeze(1).unsqueeze(0)
	kernal = torch.max(torch.abs((grid_pts - center))*z_scaling.unsqueeze(0).unsqueeze(1).unsqueeze(2), dim=-1)[0].reshape(11,11,10*5+1)
	kernal = torch.ceil(kernal)
	if use_cuda:
		print ("Using CUDA")
		kernal = kernal.cuda()
	return kernal

proposal_network = UNet4(4).cuda()
proposal_network = nn.DataParallel(proposal_network)
proposal_network.eval()
pytorch_total_params = sum(p.numel() for p in proposal_network.parameters())
print ('network param: (Mb)', pytorch_total_params/1024/1024)
proposal_network.load_state_dict(torch.load(\
os.path.join('trained_checkpoints','230823-165312','model_epoch_'+str(25)+'.pt')))

def packing_scene_initialize(model):
	scene_top_down = np.zeros((
		int(model.x_max / model.voxel_resolution/5),
		int(model.y_max / model.voxel_resolution/5))).astype('int')
	scene_occ = np.zeros((
		int(model.x_max / model.voxel_resolution/5),
		int(model.y_max / model.voxel_resolution/5),
		int(model.z_max / model.voxel_resolution))).astype('bool')
	
	# Create sdf voxel
	scene_sdf = np.ones((
		int(model.x_max / model.voxel_resolution/5),
		int(model.y_max / model.voxel_resolution/5),
		int(model.z_max / model.voxel_resolution))) * 6.0
	for sdf_step in range(0,6):
		scene_sdf[
			sdf_step : scene_sdf.shape[0]-sdf_step,
			sdf_step : scene_sdf.shape[1]-sdf_step,
			sdf_step * 5: scene_sdf.shape[2]
		] = sdf_step
	return scene_top_down, scene_occ, scene_sdf

def get_convex_hull(mat):
	# Compute the convex hull of the primary object
	array_scatter = []
	## calculate points for each contour
	for si in range(mat.shape[0]):
		for sj in range(mat.shape[1]):
			if mat[si, sj]:
				array_scatter.append([[si, sj]])
	if array_scatter == []:
		return None
	else:
		array_scatter = np.array(array_scatter).astype('int32')
		conv_hull = cv2.convexHull(array_scatter, False)
		return conv_hull

def get_objective_value(scene_occ, obj_occ, i, j, k, scene_top_down, obj_top_down, scene_sdf, choose_objective='sdf', sdf_remain_terms='1234', reorder=False):
	obj_size = obj_occ.shape
	scene_size = scene_occ.shape
	c = 1
	if choose_objective == 'sdf':
		if sdf_remain_terms == '1234':
			objective_ = (
				k 
				+ 5.0/2 * (scene_sdf[i:i+obj_size[0], j:j+obj_size[1], k:k+obj_size[2]] * obj_occ).sum() / (obj_occ).sum() \
				+ 20.0/2 * (1.0 - obj_occ.sum()**(1/3) / (obj_size[0] * obj_size[1] * obj_size[2])**(1/3))
			)
			if reorder:
				objective_ += 160.0/2 * (1.0 - obj_occ.sum()**(1/3) / (scene_size[0] * scene_size[1] * scene_size[2])**(1/3))
		else: # for example, sdf_remain_terms == ['1','4']
			objective_ = 0.0
			if '1' in sdf_remain_terms:
				objective_ += k
			if '2' in sdf_remain_terms:
				objective_ += 5.0/2 * (scene_sdf[i:i+obj_size[0], j:j+obj_size[1], k:k+obj_size[2]] * obj_occ).sum() / (obj_occ).sum()
			if '3' in sdf_remain_terms:
				objective_ += 20.0/2 * (1.0 - obj_occ.sum()**(1/3) / (obj_size[0] * obj_size[1] * obj_size[2])**(1/3))
			if '4' in sdf_remain_terms:
				objective_ += 160.0/2 * (1.0 - obj_occ.sum()**(1/3) / (scene_size[0] * scene_size[1] * scene_size[2])**(1/3))
	elif choose_objective == 'dblf':
		objective_ = k + c*(i+j)
		if reorder:
			objective_ += 160.0/2 * (1.0 - obj_occ.sum()**(1/3) / (scene_size[0] * scene_size[1] * scene_size[2])**(1/3))
	elif choose_objective == 'hm':
		Hc = np.where(obj_top_down==0, scene_top_down, np.maximum(k+obj_top_down, scene_top_down))
		objective_ = np.sum(Hc) + c*(i+j)
		if reorder:
			objective_ += 160.0/2 * (1.0 - obj_occ.sum()**(1/3) / (scene_size[0] * scene_size[1] * scene_size[2])**(1/3))
	elif choose_objective == 'mta':
		touching_ = 0.0
		if i == 0:
			touching_ += np.count_nonzero(obj_occ[0,:,:])
		else:
			touching_ += np.count_nonzero(np.any(\
				np.logical_and(scene_occ[i-1:i+obj_size[0]-1, j:j+obj_size[1], k:k+obj_size[2]], obj_occ), 2))
		if i >= scene_size[0]-obj_size[0]:
			touching_ += np.count_nonzero(obj_occ[-1,:,:])
		else:
			touching_ += np.count_nonzero(np.any(\
				np.logical_and(scene_occ[i+1:i+obj_size[0]+1, j:j+obj_size[1], k:k+obj_size[2]], obj_occ), 2))
		if j == 0:
			touching_ += np.count_nonzero(obj_occ[:,0,:])
		else:
			touching_ += np.count_nonzero(np.any(\
				np.logical_and(scene_occ[i:i+obj_size[0], j-1:j+obj_size[1]-1, k:k+obj_size[2]], obj_occ), 2))
		if j >= scene_size[1]-obj_size[1]:
			touching_ += np.count_nonzero(obj_occ[:,-1,:])
		else:
			touching_ += np.count_nonzero(np.any(\
				np.logical_and(scene_occ[i:i+obj_size[0], j+1:j+obj_size[1]+1, k:k+obj_size[2]], obj_occ), 2))
		if k < 5:
			touching_ += np.count_nonzero(obj_occ[:,:,0:5])
		else:
			touching_ += np.count_nonzero(np.any(\
				np.logical_and(scene_occ[i:i+obj_size[0], j:j+obj_size[1], k-5:k+obj_size[2]-5], obj_occ), 2))
		if k >= scene_size[2]-obj_size[2]-4:
			touching_ += 0
		else:
			touching_ += np.count_nonzero(np.any(\
				np.logical_and(scene_occ[i:i+obj_size[0], j:j+obj_size[1], k+5:k+obj_size[2]+5], obj_occ), 2))
		objective_ = touching_ 
		if reorder:
			objective_ -= 160.0/2 * (1.0 - obj_occ.sum()**(1/3) / (scene_size[0] * scene_size[1] * scene_size[2])**(1/3))
	if choose_objective == 'sdf':
		return objective_, \
			k, \
			5.0/2 * (scene_sdf[i:i+obj_size[0], j:j+obj_size[1], k:k+obj_size[2]] * obj_occ).sum() / (obj_occ).sum(), \
			20.0/2 * (1.0 - obj_occ.sum()**(1/3) / (obj_size[0] * obj_size[1] * obj_size[2])**(1/3)), \
			160.0/2 * (1.0 - obj_occ.sum()**(1/3) / (scene_size[0] * scene_size[1] * scene_size[2])**(1/3))
	else:
		return objective_

def get_scene_object_heightmaps(scene_occ, obj_occ, i, j):
	obj_size = obj_occ.shape
	scene_size = scene_occ.shape
	scene_top_down = scene_size[2] - np.argmax(np.flip(scene_occ[i:i+obj_size[0], j:j+obj_size[1]], 2),2)\
		 - np.logical_not(np.any(scene_occ[i:i+obj_size[0], j:j+obj_size[1]], 2)) * scene_size[2]
	obj_top_down = obj_size[2] - np.argmax(np.flip(obj_occ, 2),2) - np.logical_not(np.any(obj_occ, 2)) * obj_size[2]
	obj_bottom_up = np.argmax(obj_occ, 2) + np.logical_not(np.any(obj_occ, 2)) * 1000 # * obj_size[2]
	return scene_top_down, obj_top_down, obj_bottom_up

def get_current_scene_state(model):
	import matplotlib.pyplot as plt
	viewMatrix = p.computeViewMatrix(cameraEyePosition=[0.5, 0, 3.1],cameraTargetPosition=[0.5, 0, 0],cameraUpVector=[1, 0, 0])
	projectionMatrix = p.computeProjectionMatrixFOV(fov=10.0,aspect=1.0,nearVal=0.1,farVal=3.2)
	width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=54, height=54,viewMatrix=viewMatrix,projectionMatrix=projectionMatrix)
	depthImg2 = depthImg[11:54-11,12:54-10]
	from PIL import Image
	import PIL
	# create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
	proj_matrix = np.asarray(projectionMatrix).reshape([4, 4], order="F")
	view_matrix = np.asarray(viewMatrix).reshape([4, 4], order="F")
	tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

	# create a grid with pixel coordinates and depth values
	y, x = np.mgrid[-1:1:2 / depthImg2.shape[0], -1:1:2 / depthImg2.shape[1]]
	y *= -1.
	x, y, z = x.reshape(-1), y.reshape(-1), depthImg2.reshape(-1)
	h = np.ones_like(z)

	pixels = np.stack([x, y, z, h], axis=1)
	pixels[:, 2] = 2 * pixels[:, 2] - 1

	# turn pixels to world coordinates
	points = np.matmul(tran_pix_world, pixels.T).T
	points /= points[:, 3: 4]
	points = points[:, :3]
	scene_top_down = np.zeros(depthImg2.shape)
	for i in range(depthImg2.shape[0]):
		for j in range(depthImg2.shape[1]):
			scene_top_down[i,j] = np.round((points[depthImg2.shape[0]*i+j,2])/model.voxel_resolution*100.0)
	scene_top_down = np.asarray(Image.fromarray(scene_top_down).transpose(PIL.Image.FLIP_LEFT_RIGHT).transpose(PIL.Image.FLIP_TOP_BOTTOM))
	scene_top_down = scene_top_down.astype('int')
	return scene_top_down

def update_scene_occ(last_scene_top_down, scene_top_down, scene_occ):
	for i_ in range(scene_occ.shape[0]):
		for j_ in range(scene_occ.shape[1]):
			if last_scene_top_down[i_,j_] != scene_top_down[i_, j_]:
				scene_occ[i_,j_,0:scene_top_down[i_,j_]] = True
	return scene_top_down, scene_occ

def construct_sdf(scene_occ, kernal, use_cuda):
	conv_scene_occ = np.pad(scene_occ, ((5,5),(5,5),(25,25)), 'constant', constant_values=((1,1),(1,1),(1,0)))
	k_i = k_j = kernal.shape[0]
	k_k = kernal.shape[2]
	if use_cuda:
		input_ = torch.tensor(conv_scene_occ).float().cuda()
		# First slice the horizontal planes into patches of k_i*k_i, and view the z dimention as channel (i.e., what input_.permute(2,0,1).unsqueeze(0) did)
		slices = F.unfold(input_.permute(2,0,1).unsqueeze(0), kernel_size=k_i, dilation=1, stride=1).permute(0,2,1).view(1,input_.shape[0]-k_i+1,input_.shape[1]-k_i+1,input_.shape[2],k_i,k_i)
		#intermediate = F.unfold(input.permute(2,0,1).unsqueeze(0), kernel_size=11, dilation=1, stride=1).permute(0,2,1).view(1,32,32,200,11,11)
		# Second, slice the vertical dimension
		slices = slices.reshape((input_.shape[0]-k_i+1)*(input_.shape[1]-k_i+1),input_.shape[2],k_i*k_i).unfold(dimension=1,size=k_k,step=1).reshape(input_.shape[0]-k_i+1 ,input_.shape[1]-k_i+1 ,input_.shape[2]-k_k+1,k_i,k_i,k_k)
		scene_sdf = torch.zeros(scene_occ.shape).cuda()
		x_part_size = int(scene_occ.shape[0]/2)
		for si in range(2):
			values = (slices[si*x_part_size:(si+1)*x_part_size].reshape(x_part_size*scene_occ.shape[1]*scene_occ.shape[2],k_i, k_j, k_k) * kernal).reshape(x_part_size, scene_occ.shape[1]*scene_occ.shape[2], k_i*k_j*k_k)
			values = (values == 0)*5.0+values
			scene_sdf[si*x_part_size:(si+1)*x_part_size] = torch.min(values, dim=-1)[0].reshape(x_part_size,scene_occ.shape[1],scene_occ.shape[2])
			torch.cuda.empty_cache()
		scene_sdf = scene_sdf.cpu().numpy()
	else:
		#slices = [conv_scene_occ[i:i+k_i,j:j+k_j,k:k+k_k] for i in range(conv_scene_occ.shape[0]-k_i+1) for j in range(conv_scene_occ.shape[1]-k_j+1) for k in range(conv_scene_occ.shape[2]-k_k+1)]
		input_ = torch.tensor(conv_scene_occ).float()
		# First slice the horizontal planes into patches of k_i*k_i, and view the z dimention as channel (i.e., what input_.permute(2,0,1).unsqueeze(0) did)
		slices = F.unfold(input_.permute(2,0,1).unsqueeze(0), kernel_size=k_i, dilation=1, stride=1).permute(0,2,1).view(1,input_.shape[0]-k_i+1,input_.shape[1]-k_i+1,input_.shape[2],k_i,k_i)
		# Second, slice the vertical dimension
		slices = slices.reshape((input_.shape[0]-k_i+1)*(input_.shape[1]-k_i+1),input_.shape[2],k_i*k_i).unfold(dimension=1,size=k_k,step=1).reshape(input_.shape[0]-k_i+1 ,input_.shape[1]-k_i+1 ,input_.shape[2]-k_k+1,k_i,k_i,k_k)
		values = (slices.reshape(scene_occ.shape[0]*scene_occ.shape[1]*scene_occ.shape[2],k_i, k_j, k_k) * kernal).reshape(scene_occ.shape[0]*scene_occ.shape[1]*scene_occ.shape[2], k_i*k_j*k_k)
		values = (values == 0)*5.0+values
		scene_sdf = torch.min(values, dim=-1)[0].reshape(scene_occ.shape).numpy()
	return scene_sdf

def get_method_name(choose_objective):
	if choose_objective == 'sdf':
		mtd_str_ = 'SDFM'
	elif choose_objective == 'dblf':
		mtd_str_ = 'DBLF'
	elif choose_objective == 'hm':
		mtd_str_ = 'HM'
	elif choose_objective == 'mta':
		mtd_str_ = 'MTA'
	elif choose_objective == 'random':
		mtd_str_ = 'RANDOM'
	elif choose_objective == 'first':
		mtd_str_ = 'FIRST-FIT'
	elif choose_objective == 'genetic':
		mtd_str_ = 'GENETIC-DBLF'
	return mtd_str_

def pack_an_object(model, env, shape_codes, best_obj_idx, best_r_id, best_i, best_j, best_k, dict_obj_occ, list_obj_occ, packed_volumes, is_simulate=True):
	obj_occ = dict_obj_occ[str(best_obj_idx)+'_r'+str(best_r_id)]
	obj_size = obj_occ.shape
	list_obj_occ.append({'start_pos':[best_i,best_j,best_k], 'obj_occ':obj_occ, 'obj_idx':best_obj_idx})
	# 2. Place the couple/object.
	tmp_obj1 = copy.deepcopy(model.obj_list[best_obj_idx]).apply_transform(rotation_matrix(best_r_id*np.pi/4, [0, 0, 1]))
	model.obj_pos[best_obj_idx][0] = -model.x_max/2 - tmp_obj1.bounds[0][0] + best_i*model.voxel_resolution*5
	model.obj_pos[best_obj_idx][1] = -model.y_max/2 - tmp_obj1.bounds[0][1] + best_j*model.voxel_resolution*5
	model.obj_pos[best_obj_idx][2] = -model.z_max - tmp_obj1.bounds[0][2] + best_k*model.voxel_resolution
	obj_filename = os.path.join('./autostore/models/our_oriented_dataset/', np.sort(os.listdir('./autostore/models/our_oriented_dataset/'))[shape_codes[best_obj_idx]])
	obj_file_basename = os.path.basename(obj_filename)
	model.obj_rot[best_obj_idx] = best_r_id * np.pi / 4
	obs, reward, _, info = env.insert_a_packing_object(obj_file_basename[:-4], \
		([0.5+model.obj_pos[best_obj_idx][0].numpy()/100.0, model.obj_pos[best_obj_idx][1].numpy()/100.0, (model.obj_pos[best_obj_idx][2].numpy()+model.z_max)/100.0+0.1], \
		p.getQuaternionFromEuler([0,0,model.obj_rot[best_obj_idx][0].numpy().tolist()])))#, color=[np.random.rand(), np.random.rand(),np.random.rand(),0])#[1*(obj_idx/len(model.obj_list)), 0, 1*(1 - obj_idx/len(model.obj_list)), 0.5])
	p.changeDynamics(env.obj_ids['rigid'][-1], -1, mass=model.obj_list[best_obj_idx].volume*10)
	packed_volumes.append(obj_occ.sum()/5)#.append(model.obj_list[best_obj_idx].volume)
	if is_simulate:
		for _ in range(180):
			p.stepSimulation()
	return list_obj_occ, packed_volumes

def update_objective_maps(scene_top_down, scene_occ, scene_sdf, dict_obj_occ, obj_idx, choose_objective, require_stability, dict_objective, update_region=None, sdf_remain_terms='1234', update_local=False, reorder=False, feasible_positions=None):
	# Specify the invalid values and infalid height for each method.
	best_obj_idx, best_i, best_j, best_k, best_r_id = 0, 0, 0, 0, 0
	if choose_objective == 'mta':
		invalid_value = -1
	elif choose_objective in ['dblf', 'hm', 'sdf', 'random', 'first']:
		invalid_value = 1000000.0
	invalid_height = 1000000.0

	for r in range(4):
		# Get basic size information of the bin and the object
		obj_occ = dict_obj_occ[str(obj_idx)+'_r'+str(r)]
		scene_size = scene_occ.shape
		obj_size = obj_occ.shape
		
		# If the object is too large to fit the bin at the current orientation, then skip.
		if (np.array(obj_size) > np.array(scene_size)).any():
			continue
		
		# Initialize the objective maps. 
		# Note that local update scheme is only used in ['dblf', 'mta', 'hm', 'sdf'].
		if choose_objective in ['random']: 
			# If the mode is random/first-fit, re-initialize the whole objective map.
			objective_map = np.ones((scene_size[0]-obj_size[0]+1, scene_size[1]-obj_size[1]+1, 2)) * invalid_value
			objective_map[:,:,1] = invalid_height
		elif choose_objective in ['dblf', 'mta', 'hm', 'first']:
			# If the objective map exists in the dictionary load the map; otherwise, initialize the map.
			if str(obj_idx)+'_r'+str(r)+'_map' in dict_objective.keys():
				objective_map = dict_objective[str(obj_idx)+'_r'+str(r)+'_map']
				just_initialized = False
			else:
				objective_map = np.ones((scene_size[0]-obj_size[0]+1, scene_size[1]-obj_size[1]+1, 2)) * invalid_value
				objective_map[:,:,1] = invalid_height
				just_initialized = True
		elif choose_objective == 'sdf':
			if str(obj_idx)+'_r'+str(r)+'_map' in dict_objective.keys():
				objective_map = dict_objective[str(obj_idx)+'_r'+str(r)+'_map']
				just_initialized = False
			else:
				objective_map = np.ones((scene_size[0]-obj_size[0]+1, scene_size[1]-obj_size[1]+1, 5)) * invalid_value
				objective_map[:,:,1] = invalid_height
				just_initialized = True

		# Speicify the searching region.
		# Note that local update scheme is only used in ['dblf', 'mta', 'hm', 'sdf'].
		if choose_objective == 'random':
			search_i = list(np.random.randint(0, scene_size[0]-obj_size[0]+1, 3))
			search_j = list(np.random.randint(0, scene_size[1]-obj_size[1]+1, 3))
		elif choose_objective in ['dblf', 'mta', 'hm', 'sdf', 'first']:
			# If there is not a specific update region, or the map is just initialized, update the whole map
			if just_initialized:
				update_region = [0, 0, scene_size[0], scene_size[1]]
			elif (scene_top_down==dict_objective[str(obj_idx)+'_r'+str(r)+'_sceneTD']).all():
				update_region = [0,0,0,0]
			else:
				np_where_0, np_where_1 = np.where(scene_top_down!=dict_objective[str(obj_idx)+'_r'+str(r)+'_sceneTD'])
				update_region = [min(np_where_0), min(np_where_1), max(np_where_0), max(np_where_1)]
				#print ('update_region', update_region)
			search_i = list(range(max(0, update_region[0]-obj_size[0]+1), min(scene_size[0]-obj_size[0]+1, update_region[2]+obj_size[0]), 1))
			search_j = list(range(max(0, update_region[1]-obj_size[1]+1), min(scene_size[1]-obj_size[1]+1, update_region[3]+obj_size[1]), 1))

		# If stability is required to measure, compute the mass center of object
		if require_stability:
			nonzeros = np.nonzero(obj_occ)
			tmp_obj_center = np.array([np.mean(nonzeros[0]), np.mean(nonzeros[1]), np.mean(nonzeros[2])])

		for i in search_i:
			for j in search_j:
				# ********************TRY TO USE NETWORK TO SPEED UP SDF-PACK ***************
				if True: #choose_objective == 'sdf':
					netout_pos_i = i + int(np.floor((obj_size[0]-1)/2))
					netout_pos_j = j + int(np.floor((obj_size[1]-1)/2))
					if (feasible_positions[r, netout_pos_i, netout_pos_j]==0):
						objective_map[i,j,0] = invalid_value
						objective_map[i,j,1] = invalid_height
						if choose_objective == 'sdf':
							objective_map[i,j,2] = invalid_value
							objective_map[i,j,3] = invalid_value
							objective_map[i,j,4] = invalid_value
						#print ('skipped')
						continue
				# ********************TRY TO USE NETWORK TO SPEED UP SDF-PACK ***************
				scene_part_top_down, obj_top_down, obj_bottom_up = get_scene_object_heightmaps(scene_occ, obj_occ, i, j)
				k = max((scene_part_top_down-obj_bottom_up).reshape(-1).tolist())
				if k < 0:
					k = 0
				if k >= scene_size[2]-obj_size[2]:
					objective_map[i,j,0] = invalid_value
					objective_map[i,j,1] = invalid_height
					if choose_objective == 'sdf':
						objective_map[i,j,2] = invalid_value
						objective_map[i,j,3] = invalid_value
						objective_map[i,j,4] = invalid_value
					continue
				# 2. Stability Check
				if not require_stability:
					is_stable = True
				else:
					is_stable = False
					if k <= 5:
						support_face = (np.any(obj_occ[:,:,0:5], axis=-1))
					else:
						support_face = (np.any(np.logical_and(\
							scene_occ[i:i+obj_size[0], j:j+obj_size[1], k-6:k-1], obj_occ[:,:,0:5]), axis=-1))
					support_polygon = get_convex_hull(support_face)
					if support_polygon is not None and cv2.pointPolygonTest(support_polygon, (tmp_obj_center[0], tmp_obj_center[1]), 1)>0.5: # It IS STABLE
						is_stable = True
				if not is_stable:
					objective_map[i,j,0] = invalid_value
					objective_map[i,j,1] = invalid_height
					if choose_objective == 'sdf':
						objective_map[i,j,2] = invalid_value
						objective_map[i,j,3] = invalid_value
						objective_map[i,j,4] = invalid_value
					continue
				# 3. Find the placement with minimal (MTA: maximal) objective to execute
				# Find the best object and placement with minimal (MTA: maximal) objective value
				if choose_objective == 'mta':
					objective_map[i,j,0] = get_objective_value(scene_occ, obj_occ, i, j, k, scene_part_top_down, obj_top_down, scene_sdf, choose_objective, reorder=reorder)
					objective_map[i,j,1] = k
				elif choose_objective in ['dblf', 'hm']:
					objective_map[i,j,0] = get_objective_value(scene_occ, obj_occ, i, j, k, scene_part_top_down, obj_top_down, scene_sdf, choose_objective, sdf_remain_terms=sdf_remain_terms, reorder=reorder)
					objective_map[i,j,1] = k
				elif choose_objective == 'sdf':
					res_ = get_objective_value(scene_occ, obj_occ, i, j, k, scene_part_top_down, obj_top_down, scene_sdf, choose_objective, sdf_remain_terms=sdf_remain_terms, reorder=reorder)
					objective_map[i,j,0] =  res_[0]
					objective_map[i,j,1] = res_[1]
					objective_map[i,j,2] = res_[2]
					objective_map[i,j,3] = res_[3]
					objective_map[i,j,4] = res_[4]
				elif choose_objective in ['random', 'first']: # If the mode is 'random' or 'first-fit', break at the first found feasible placement.
					objective_map[i,j,0] = 0
					objective_map[i,j,1] = k
					break
		dict_objective[str(obj_idx)+'_r'+str(r)+'_map'] = objective_map
		dict_objective[str(obj_idx)+'_r'+str(r)+'_sceneTD'] = scene_top_down
	return dict_objective

def heuristic_ppn_packing(args, model, kernal, shape_codes, env, dict_obj_occ, require_stability=False, choose_objective='sdf', fix_sequence_length=5, use_cuda=False, sdf_remain_terms='1234', vol_dec=False, train_data_save_dict=None):
	# This function implements bin packing for irregular objects with controllable arriving order.
	# given the model and the environment
	"""
	:param model: the packing model class, including information of each object, the location and orientation, bin size, etc.
	:param kernal: torch.tensor(2*5+1, 2*5+1, 2*5*5+1) [Could cuda if use_cuda=True]. For PyTorch fast computation of the SDF field.
	:param shape_codes: []. The ids for all objects to be packed.
	:param env: the PyBullet environment class, including the robot arms, the bin, the physical simulation etc.
	:param dict_obj_occ: pre-scanned object heightmaps and pre-constructed object occ.
	:param require_stability: True if only stable placements are valid. The stability is measured by if the object's mass center is located inside the support polygon corresponding to a target location.
	:param choose_objective: 'sdf', 'dblf', 'hm', 'mta', 'random' or 'first'
	:param fix_sequence_length: length of the packing buffer. Objects packing order is controllable in the buffer. Once the buffer is full, packing ends.
	:param use_cuda: set as 'True' to use CUDA computation to speed up the SDF construction.
	"""
	# Define Method Names
	mtd_str_ = get_method_name(choose_objective)
	
	# Initialize Timers
	reorder_tot_time = 0.0
	left_list = []
	list_obj_occ = []
	packed_volumes = []

	# Define packing list
	obj_idx = 0
	list_to_place = np.arange(len(model.obj_list)).astype('int').tolist()

	# Create scene voxel
	last_scene_top_down, scene_occ, scene_sdf = packing_scene_initialize(model)
	
	# Initialize a 2D objective map for each object
	dict_objective = {}
	
	# Find the best object and the best position to place
	while list_to_place != []:
		# Initialize
		success_ = False
		c = 1 # weighting for x and y location
		if choose_objective == 'mta':
			max_objective = 0.0
		elif choose_objective in ['dblf', 'hm', 'sdf', 'random', 'first']:
			min_objective = 1000000.0
		
		scene_top_down = get_current_scene_state(model)
		#hr_scene = get_high_res_current_scene_state(container_height=0.30)
		#scene_top_down = np.ceil(np.asarray(Image.fromarray(hr_scene).resize((32,32), Image.BILINEAR))*100.0).astype('int')
		_, scene_occ = update_scene_occ(last_scene_top_down, scene_top_down, scene_occ)
		
		obj_start_ = time.time()
		searching_list = list_to_place[0:fix_sequence_length]

		if vol_dec:
			searching_list = sorted(searching_list, key=lambda coord:model.obj_bbox_volume_list[coord], reverse=True)
			
		if choose_objective == 'sdf':
			update_local=True
		else:
			update_local=False

		# ********************TRY TO USE NETWORK TO SPEED UP SDF-PACK ***************
		scene_size = scene_top_down.shape
		if args.accelerate=='network':
			inputs_ = torch.tensor([]).cuda()
			for obj_idx in searching_list:
				for r in range(1):
					_, obj_top_down_, obj_bottom_up_ = get_scene_object_heightmaps(np.zeros((scene_size[0],scene_size[1],150)), dict_obj_occ[str(obj_idx)+'_r'+str(r)], 0,0)
					obj_top_down_ = obj_top_down_.astype('float32')
					obj_bottom_up_ = obj_bottom_up_.astype('float32') / 5 * 0.01 / 0.3
					obj_top_down_ = obj_top_down_ / 5 * 0.01 / 0.3
					obj_bottom_up_[obj_bottom_up_ == obj_bottom_up_.max()] = 1.0
					if obj_top_down_.shape[0] > scene_size[0] or obj_top_down_.shape[1] > scene_size[1]:
						obj_top_down = np.ones((scene_size[0],scene_size[1]))
						obj_bottom_up = np.zeros((scene_size[0],scene_size[1]))
					else:
						obj_top_down = np.pad(obj_top_down_, \
								((int(np.floor((scene_size[0]-obj_top_down_.shape[0])/2)), int(np.ceil((scene_size[0]-obj_top_down_.shape[0])/2))), \
								(int(np.floor((scene_size[1]-obj_top_down_.shape[1])/2)), int(np.ceil((scene_size[1]-obj_top_down_.shape[1])/2)))), \
								'constant', constant_values=(0,0)).astype('float32')
						obj_bottom_up = np.pad(obj_bottom_up_, \
								((int(np.floor((scene_size[0]-obj_bottom_up_.shape[0])/2)), int(np.ceil((scene_size[0]-obj_bottom_up_.shape[0])/2))), \
								(int(np.floor((scene_size[1]-obj_bottom_up_.shape[1])/2)), int(np.ceil((scene_size[1]-obj_bottom_up_.shape[1])/2)))), \
								'constant', constant_values=(1,1)).astype('float32')
					scene_top_down_ = scene_top_down/0.35*255.0
					inputs_ = torch.cat((inputs_, torch.cat((torch.tensor(scene_top_down_).unsqueeze(0).unsqueeze(1), \
						torch.tensor(obj_top_down).unsqueeze(0).unsqueeze(1), \
						torch.tensor(obj_bottom_up).unsqueeze(0).unsqueeze(1)), dim=1).float().cuda()), dim=0)
			st__ = time.time()
			outscore = proposal_network(inputs_).detach()
			print ('network forward time:', time.time() - st__)
			score_top_50 = torch.topk(outscore.reshape(outscore.shape[0], -1), args.topk*4, largest=False, dim=-1)[0][:,-1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
			feasible_positions = (outscore<score_top_50).cpu().numpy() #torch.logical_or(outclass[:,0,:,:]>0.4, outscore<score_top_200).cpu().numpy()
		elif args.accelerate=='plain':
			feasible_positions = torch.ones((len(searching_list),4,32,32))
		# ********************TRY TO USE NETWORK TO SPEED UP SDF-PACK ***************
		# Find the best object and placement with minimal (MTA: maximal) objective value
		tot_binary_acc = 0.0
		tot_binary_cnt = 0.0
		for obj_idx in searching_list:
			in_feasible_positions = feasible_positions[searching_list.index(obj_idx)]
			if vol_dec:
				dict_objective = update_objective_maps(scene_top_down, scene_occ, scene_sdf, dict_obj_occ, obj_idx, choose_objective, require_stability, dict_objective, sdf_remain_terms=sdf_remain_terms, update_local = update_local, reorder=False, feasible_positions = in_feasible_positions)
			else:
				dict_objective = update_objective_maps(scene_top_down, scene_occ, scene_sdf, dict_obj_occ, obj_idx, choose_objective, require_stability, dict_objective, sdf_remain_terms=sdf_remain_terms, update_local = update_local, reorder=True, feasible_positions = in_feasible_positions)
			for r in range(4):
				# Only if the couple fails to get placed, decouple it, and consider placing the primary object as an individual object
				obj_occ = dict_obj_occ[str(obj_idx)+'_r'+str(r)]
				if str(obj_idx)+'_r'+str(r)+'_map' not in dict_objective:
					continue
				objective_map = dict_objective[str(obj_idx)+'_r'+str(r)+'_map']
				obj_size = obj_occ.shape
				# Find the best object and placement with minimal (MTA: maximal) objective value
				if choose_objective == 'mta':
					if np.max(objective_map[...,0]) > max_objective:
						max_objective = np.max(objective_map[...,0])
						argmin_ = np.argmax(objective_map[...,0])
						best_i = int(np.floor(argmin_/objective_map.shape[1]))
						best_j = int(argmin_%objective_map.shape[1])
						best_k = int(objective_map[best_i, best_j, 1])
						best_r_id = r
						best_obj_idx = obj_idx
						success_ = True
				elif choose_objective in ['dblf', 'hm', 'sdf', 'random', 'first']:
					if np.min(objective_map[...,0]) < min_objective:
						min_objective = np.min(objective_map[...,0])
						argmin_ = np.argmin(objective_map[...,0])
						best_i = int(np.floor(argmin_/objective_map.shape[1]))
						best_j = int(argmin_%objective_map.shape[1])
						best_k = int(objective_map[best_i, best_j, 1])
						best_r_id = r
						best_obj_idx = obj_idx
						success_ = True
			if vol_dec and success_:
				break
		# If the best object and the best placement is found
		if success_:
			list_obj_info = []
			for tmp_obj_idx in searching_list:
				for tmp_obj_r in range(4):
					key = str(tmp_obj_idx)+'_r'+str(tmp_obj_r)+'_map'
					if key not in dict_objective.keys():
						continue
					obj_map = dict_objective[key]
					obj_map = np.pad(obj_map, \
						((int(np.floor((32-obj_map.shape[0])/2)), int(np.ceil((32-obj_map.shape[0])/2))), \
						(int(np.floor((32-obj_map.shape[1])/2)), int(np.ceil((32-obj_map.shape[1])/2))), (0,0)), \
						'constant', constant_values=(1000,1000))
					list_obj_info.append({'shape_type':shape_codes[tmp_obj_idx], 'r':tmp_obj_r, 'objective_map':obj_map})
			reorder_tot_time += time.time() - obj_start_
			
			# 1. Pack the object
			if vol_dec:
				print (mtd_str_+' vol-dec placement SUCCESS. Placed item', best_obj_idx, 'Best', (best_i, best_j, best_k, best_r_id))
			else:
				print (mtd_str_+' re-order placement SUCCESS. Placed item', best_obj_idx, 'Best', (best_i, best_j, best_k, best_r_id))
			list_to_place.remove(best_obj_idx)
			train_data_save_dict.append(shape_codes[best_obj_idx])
			list_obj_occ, packed_volumes = pack_an_object(model, env, shape_codes, best_obj_idx, best_r_id, best_i, best_j, best_k, dict_obj_occ, list_obj_occ, packed_volumes)
			
			# 2. Update the scene occupancy and scene sdf
			#    A Local Update Scheme: update only regions around the newly placed object
			update_start_ = time.time()
			last_scene_top_down = scene_top_down
			if choose_objective == 'sdf':
				scene_sdf = construct_sdf(scene_occ, kernal, use_cuda)
			
			# 3. Update the 2D objective maps for the remaining (unplaced) objects (remain_obj_idx)
			#    A Local Update Scheme: update only regions around the newly placed object
			reorder_tot_time += time.time() - update_start_
		else:
			reorder_tot_time += time.time() - obj_start_
			decouple_start_ = time.time()
			reorder_tot_time += time.time() - decouple_start_
			# If the buffer is full and the remaining objects cannot be placed
			# Stop the packing algorithm and quit.
			if vol_dec:
				print (mtd_str_+' vol-dec placement FAIL. Unable to place items', list_to_place)
			else:
				print (mtd_str_+' re-order placement FAIL. Unable to place items', list_to_place)
			break

	print ('Stability Required:', require_stability)
	if choose_objective == 'sdf' and sdf_remain_terms != '1234':
		print ('SDF remain terms:', sdf_remain_terms)
	if fix_sequence_length != 5:
		print ('with sequence length', fix_sequence_length)
	print ('Time for '+mtd_str_+'re-order placement:', reorder_tot_time)

	return scene_occ, scene_sdf, list_obj_occ, reorder_tot_time, packed_volumes, last_scene_top_down, train_data_save_dict
