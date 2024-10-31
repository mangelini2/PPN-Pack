## PPN-Pack: Placement Proposal Network for Efficient Robotic Bin Packing

This repository is for our paper *'[PPN-Pack: Placement Proposal Network for Efficient Robotic Bin Packing](https://ieeexplore.ieee.org/document/10493124)'* published in IEEE Robotics and Automation Letters (RA-L), 2024. In this work, we introduce _PPN-Pack_, a novel learning-based approach to improve the efficiency of packing general objects. Our method works together with a packing heuristic, and can more quickly locate the compact packing placements while achieving similar packing compactness as the upper-bound ([_SDF-Minimization_](https://github.com/kwpoon/SDF-Pack)).


<h4 align = "center"> Visual Comparison of SDF-Minimization and PPN-Pack in Packing 25 Objects</center></h4>

| SDF-Minimization <br> (Avg 1.86 s/object) | PPN-Pack <br>(Avg 0.43 s/object) | | SDF-Minimization <br>(Avg 1.86 s/object) | PPN-Pack <br>(Avg 0.43 s/object) |
| ------------- | ------------- | -------------| ------------- | ------------- |
| <video autoplay src="https://github.com/user-attachments/assets/1637ebd4-4f5f-4607-a012-ed1685de162d" > | <video autoplay src="https://github.com/user-attachments/assets/2a22486b-ae6a-4c9a-82c9-90081355fa3f">| | <video autoplay src="https://github.com/user-attachments/assets/04b405bd-f909-4cca-af12-a89877be29b8"> | <video autoplay src="https://github.com/user-attachments/assets/ca088e2e-b42d-4fcd-8c7b-76e180d16eeb">|

Our key idea is to learn to predict good placement locations for compact object packing to prune the search space and reduce packing computation. Specifically, we formulate the learning of placement proposals as a ranking task and construct a ranking loss based on the Hinge loss to rank the potentially compact placements. To enhance the placement learning, we further design a multi-resolution cross-correlation module to learn the placement compactness between the container and objects. The framework of our method is shown below.

<div style="text-align: center;">
    <img style="border-radius: 0.3125em;
    width: 98%;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=./figures/framework.png alt="">
    <br>
</div>

<!---<h4 align = "center">Packing Results of PPN-Pack Packing 25 Objects</center></h4>

| <video autoplay src="https://github.com/user-attachments/assets/021f1bd8-6d24-41a5-b431-09814f23bca9"> | <video autoplay src="https://github.com/user-attachments/assets/f856a76e-ee94-42e8-8da3-d25195bdea94">  | <video autoplay src="https://github.com/user-attachments/assets/1ba45d79-626f-4699-b415-9ca7330d04c3"> | <video autoplay src="https://github.com/user-attachments/assets/b77f063e-0f66-44e3-b924-045b72be0d7a">  | 
| ------------- | ------------- | -------------| ------------- |
<video autoplay src="https://github.com/user-attachments/assets/0ebd8ae3-d57b-4125-a306-e8c09226d33d">  |<video autoplay src="https://github.com/user-attachments/assets/0c47d438-7a13-4604-ae74-c07516dc1d10"> | <video autoplay src="https://github.com/user-attachments/assets/ab597528-7db2-4464-b8df-e758c8430fa6"> | <video autoplay src="https://github.com/user-attachments/assets/1b2f5b54-a759-4175-8159-db65e5a26022"> |-->

### Installation

* **Environmental Setup**
  ```
  conda env create -f environment.yml
  conda activate sdf_pack
  ```

### Packing Evaluation
* **Download the Object Models**

Following [SDF-Pack](https://github.com/kwpoon/SDF-Pack), we performed experiments on 96 types of objects collected from the YCB dataset and the Rutgers APC RGB-D dataset. Please download the [object models](https://drive.google.com/file/d/18x_Gtq_xMEGpZx-Nh-5N5obEjguBkbp3/view?usp=sharing) and extract to form `./autostore`. 
[comment]: <> Also, download the pre-extracted [object heightmaps](https://drive.google.com/file/d/1QU7-RJbG0uTyDw4cyklMRc-tdXqCUTkL/view?usp=sharing) and extract to form './train_data_ppn_96_dataset'.

The evaluation is performed on [2000 novel sequences](sequential_testing_shape_codes_96_type_2000_80_num_rand.npy).

* **Download PPN-Net's Checkpoint**

Please download the [trained_model](https://drive.google.com/file/d/150NGHDMdd8cE7W166vUYhPdTBxtxbCEd/view?usp=sharing) and extract the files to form the folder './trained_checkpoints'

* **Sequential Packing Evaluation**

The code will perform sequential packing, loading the objects in each testing sequence one by one. Physical simulation is performed after each packing step until the object is stabilised.

Running without visualization would make it faster to go through all packing cases. For this setting, please use the command:
```
python demos_network.py
```

Otherwise, if you want to visualize the results, you run with
```
python demos_network.py --disp=True
```

In addition, if you want to disable our PPN-Net and test the computation speed of conventional heuristics, you could try
```
python demos_network.py --disp=True --accelerate=plain
```


### Citation

```
@article{pan2024ppn,
  title={PPN-Pack: Placement Proposal Network for Efficient Robotic Bin Packing},
  author={Pan, Jia-Hui and Gao, Xiaojie and Hui, Ka-Hei and Zhu, Shize and Liu, Yun-Hui and Heng, Pheng-Ann and Fu, Chi-Wing},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```

### Acknowledgments

In this project, we use (parts of) the official implementations of the following works:

* [Ravens-Transporter Networks](https://github.com/google-research/ravens) (Physical Simulation)
* [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) (Network Architecture)

We thank the respective authors for their great work!

