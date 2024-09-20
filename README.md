<h2 align = "center"> PPN-Pack: Placement Proposal Network for Efficient Robotic Bin Packing</center></h2>
<h4 align = "center">Jia-Hui Pan<sup>1</sup>, Xiaojie Gao<sup>1,3</sup>, Ka-Hei Hui<sup>1</sup>, Shize Zhu<sup>3</sup>, Yun-Hui Liu<sup>2,3</sup>, Pheng-Ann Heng<sup>1</sup> and Chi-Wing Fu<sup>1</sup></h4>
<h4 align = "center"> <sup>1</sup>Department of Computer Science and Engineering</center>, <sup>2</sup>Department of Mechanical and Automation Engineering, </h4>
<h4 align = "center"> The Chinese University of Hong Kong.</center></h4>
<h4 align = "center"> <sup>3</sup>Hong Kong Centre for Logistics Robotics</center></h4>

https://drive.google.com/file/d/14buqq_wEJha_u8vb5SvtilHob1ZddY5D/view?usp=sharing


### Introduction

This repository is for our paper *'PPN-Pack: Placement Proposal Network for Efficient Robotic Bin Packing'* published in IEEE Robotics and Automation Letters (RA-L), 2024. In this work, we introduce PPNPack, a novel learning-based approach to improve the efficiency of packing general objects. Our key idea is to learn to predict good placement locations for compact object packing to prune the search space and reduce packing computation. Specifically, we formulate the learning of placement proposals as a ranking task and construct a ranking loss based on the Hinge loss to rank the potentially compact placements. To enhance the placement learning, we further design a multi-resolution cross-correlation module to learn the placement compactness between the container and objects.

<div style="text-align: center;">
    <img style="border-radius: 0.3125em;
    width: 98%;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=./figures/framework.png alt="">
    <br>
</div>

Our method works together with a packing heuristic (SDF-Pack in our implementation), and can more quickly locate the compact packing placements while achieving compact packing results similar to the upper-bound.

| SDF-Pack (Avg 1.86 s/object) | PPN-Pack (Avg 0.43 s/object) | | SDF-Pack (Avg 1.86 s/object) | PPN-Pack (Avg 0.43 s/object) |
| ------------- | ------------- | -------------| ------------- | ------------- |
| <video autoplay src="https://github.com/user-attachments/assets/1637ebd4-4f5f-4607-a012-ed1685de162d" > | <video autoplay src="https://github.com/user-attachments/assets/2a22486b-ae6a-4c9a-82c9-90081355fa3f">| | <video autoplay src="https://github.com/user-attachments/assets/04b405bd-f909-4cca-af12-a89877be29b8"> | <video autoplay src="https://github.com/user-attachments/assets/ca088e2e-b42d-4fcd-8c7b-76e180d16eeb">|


### Download the Object Models
Following [SDF-Pack](https://github.com/kwpoon/SDF-Pack), we performed experiments on 1000 packing sequences of 96 types of objects collected from the YCB dataset and the Rutgers APC RGB-D dataset. Please download the processed dataset from [Google Drive](https://drive.google.com/file/d/1i2iPqhWSmGWMJC3wa9Y_fVD3HyuklFAO/view?usp=sharing) and extract the files in the folder `./dataset/`. The object IDs forming the packing sequences can be found at `1000_packing_sequences_of_80_objects.npy`, which is then used to form the training data. Note that the evaluation is performed on novel sequences formed by randomly drawn objects.

```
|-- 1000_packing_sequences_of_80_objects.npy
|-- dataset  
|   |-- our_oriented_dataset
|   |   |-- 00000003_072-c_toy_airplane-processed.ply
|   |   |...
|   |-- our_oriented_decomp
|   |   |-- 00000003_072-c_toy_airplane-processed.obj
|   |   |...
|   |-- our_oriented_occs
|   |   |-- 00002777_cheezit_big_original-processed_objocc.npy
|   |   |-- 00002777_cheezit_big_original-processed_depth.npy
|   |   |...
```
The subfolder `./dataset/our_oriented_dataset/` contains the simplified object meshes processed to be watertight. These meshes are further processed through V-HACD convex decomposition for collision simulation, and the processed collision models are presented in the folder `./dataset/our_oriented_decomp/`. We also provide the voxelization results of the objects in `./dataset/our_oriented_occs/`. 

### Download the Training Dataset
To construct the training ground-truth, we run the [SDF-Pack](https://github.com/kwpoon/SDF-Pack) on 1000 packing sequences (object IDs in `1000_packing_sequences_of_80_objects.npy`) to form the ground-truth of our placement proposal network training.

The ground truth for each 

### Installation

* Environment
  ```
  conda env create -f environment.yml
  conda activate sdf_pack
  ```

### Todo List

* [ ] ~~Experimental datasets~~
* [ ] ~~Model Architecture~~
* [ ] ~~Testing Codes~~
* [ ] Model training

