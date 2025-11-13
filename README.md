# Two-Stage Vision System for Autonomous Aerial Imaging

This repo implements a hybrid detection-segmentation pipeline using:
- [DINO](https://github.com/IDEA-Research/DINO) for object detection
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) for segmentation

## Setup

```bash
git clone --recurse-submodules https://github.com/rmb72b/CS5404Project.git
cd CS5404Project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Download SAM checkpoint:
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
Save to: models/sam_vit_b_01ec64.pth

Download checkpoint0011_4scale.pth and save it in your project directory (this will load in pretrained weights):
https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_?usp=sharing
