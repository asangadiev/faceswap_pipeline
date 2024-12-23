# Face swapping pipeline using IP-Adapter

## Pipeline overview

- Face detection and embedding extraction via [Insight Face](https://github.com/deepinsight/insightface) library for the source images
- Face parsing via [Facer](https://github.com/FacePerceiver/facer) library to generate mask for the target image
- Diffusion inpainting pipeline from [Diffusers](https://github.com/huggingface/diffusers) + [IP Adapter](https://github.com/tencent-ailab/IP-Adapter) ControlNet
- (Optional) [GFPGAN](https://github.com/TencentARC/GFPGAN) for face enhancement

## Installation

1. Install [Insight Face](https://github.com/deepinsight/insightface):
   `pip install insightface`
3. 

## Features

- Face detection and landmark analysis
- Face parsing and segmentation
- Face restoration and enhancement
- Background upsampling (optional)
- Face alignment and cropping
- Mask generation and blending

## Dependencies

Install the following Python packages:
