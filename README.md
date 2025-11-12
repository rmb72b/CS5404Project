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

