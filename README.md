# SyncMVD

#### [paper](https://arxiv.org/pdf/2311.12891)

## Introduction

Official Pytorch implementation of the paper:
**Text-Guided Texturing by Synchronized Multi-View Diffusion**

Authors: Yuxin LIU, Minshan Xie, Hanyuan LIU, Tien-Tsin Wong

**Sync MV-Diff** can generate texture for a 3D object from a text prompt using a synchronized multi-view diffusion approach.
The method shares the denoised content among different views in each denoising step to ensure texture consistency and avoid seams and fragmentation.

<table>
  <tr>
  <td>
    <img src=assets/showcase/guitar0.jpeg_00.png width="170">
  </td>
  <td>
    <img src=assets/showcase/guitar0.gif width="170">
  </td>
  <td>
    <img src=assets/showcase/walk0.png_00.png width="170">
  </td>
  <td>
    <img src=assets/showcase/walk0.gif width="170">
  </td>
  </tr>
  <tr>
    <td>"bear playing guitar happily, snowing"</td>
    <td>"boy walking on the street"</td>
    <td>"bear playing guitar happily, snowing"</td>
    <td>"boy walking on the street"</td>
  </tr>
  <tr>
  <td>
    <img src=assets/showcase/guitar0.jpeg_00.png width="170">
  </td>
  <td>
    <img src=assets/showcase/guitar0.gif width="170">
  </td>
  <td>
    <img src=assets/showcase/walk0.png_00.png width="170">
  </td>
  <td>
    <img src=assets/showcase/walk0.gif width="170">
  </td>
  </tr>
  <tr>
    <td>"bear playing guitar happily, snowing"</td>
    <td>"boy walking on the street"</td>
    <td>"bear playing guitar happily, snowing"</td>
    <td>"boy walking on the street"</td>
  </tr>

## Installation
First clone the repository and install the basic dependencies
```bash
git clone https://github.com/username/project.git
cd project
pip install -r requirements.txt
```
Then install Pytorch3D
```bash

```
The pretrained models including [`stabilityai/stable-diffusion-2-depth`]( https://huggingface.co/stabilityai/stable-diffusion-2-depth), [`stabilityai/stable-diffusion-2-depth`]( https://huggingface.co/stabilityai/stable-diffusion-2-depth) and [`stabilityai/stable-diffusion-2-depth`]( https://huggingface.co/stabilityai/stable-diffusion-2-depth) will be downloaded automatically on demand.

## Data
The current program based on PyTorch3D library requires a input .obj mesh with .mtl material and related textures to read the original UV mapping of the object, which may require manual cleaning. Alternatively the program also support auto unwarping based on [XAtlas](https://github.com/jpcy/xatlas) to load mesh that does not met the above requirements. The program also supports loading .glb mesh, but it may not be stable as its a Pytorch3D experiment feature.

To avoid unexpected artifact, the object being textured should avoid flipped face normals and overlapping UV, and keep the number of triangle faces within around 40,000. You can try [Blender](https://www.blender.org/) for manual mesh cleaning and processing, or its python scripting for automation.

You can try out the method with the following pre-processed meshs:
- []()
- []()

## Inference
```bash
python run_experiment.py --config {your config}.yaml
```
Refer to []() for the list of arguments and settings you can adjust. You can change these settings by including them in a .yaml config file or passing the related arguments in command line; values specified in command line will overwrite those in config files.

## License
The

## Citation
```bibtex
@article{liu2023text,
  title={Text-Guided Texturing by Synchronized Multi-View Diffusion},
  author={Liu, Yuxin and Xie, Minshan and Liu, Hanyuan and Wong, Tien-Tsin},
  journal={arXiv preprint arXiv:2311.12891},
  year={2023}
}
```