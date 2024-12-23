# Face swapping pipeline using IP-Adapter

## Pipeline overview

- Face detection and embedding extraction via [Insight Face](https://github.com/deepinsight/insightface) library for the source images
- Face parsing via [Facer](https://github.com/FacePerceiver/facer) library to generate mask for the target image
- Diffusion inpainting pipeline from [Diffusers](https://github.com/huggingface/diffusers) + [IP Adapter](https://github.com/tencent-ailab/IP-Adapter) ControlNet
- (Optional) [GFPGAN](https://github.com/TencentARC/GFPGAN) for face enhancement

## Installation

1. Clone the repository:
   - `git clone https://github.com/asangt/faceswap_pipeline.git`
2. Install prerequisites by running:
   - `pip install -r requirements.txt`
3. Install [IP Adapter](https://github.com/tencent-ailab/IP-Adapter):
   - `pip install git+https://github.com/tencent-ailab/IP-Adapter.git`
4. (Optional) Install [GFPGAN](https://github.com/TencentARC/GFPGAN) for face enhancement:
   - `pip install git+https://github.com/TencentARC/GFPGAN.git`

## Required model weights

Most of the weights have embedded URLs and will be download automatically but you have to download the IP Adapter weights manually:

- IP-Adapter-FaceID-Portrait - https://huggingface.co/h94/IP-Adapter-FaceID/blob/main/ip-adapter-faceid-portrait_sd15.bin

Create and place the weights to 'ip_adapter/weights' directory or change the directory in config settings.

## How to use

For a quick start run the following after installing all the prerequisites and downloading weights:
`python diffusion_pipeline.py --source_dir <your face photos> --target_image <your target image> --output_dir <your output directory>`

You can change the used models as well as settings via the `config.py`.
