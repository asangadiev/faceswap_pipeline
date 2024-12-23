# Face swapping pipeline using IP-Adapter

## Pipeline overview

- Face detection and embedding extraction via [Insight Face](https://github.com/deepinsight/insightface) library for the source images
- Face parsing via [Facer](https://github.com/FacePerceiver/facer) library to generate mask for the target image
- Diffusion inpainting pipeline from [Diffusers](https://github.com/huggingface/diffusers)
- [IP Adapter](https://github.com/tencent-ailab/IP-Adapter) ControlNet
- [CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/h94/IP-Adapter/tree/main/models/image_encoder) - download config.json and model.safetensors, place into `ip_adapter/weights/image_encoder/` directory
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
   - The `basicsr` repository hasn't been updated for quite some time now, so with newer versions of `torchvision` you might run into the [following problem](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985)

## Required model weights

Most of the weights have embedded URLs and will be downloaded automatically but you have to download the IP Adapter weights manually:

- [IP-Adapter-FaceID-Portrait](https://huggingface.co/h94/IP-Adapter-FaceID/blob/main/ip-adapter-faceid-portrait_sd15.bin) for single reference case - place the weights to `ip_adapter/weights` directory
- [IP-Adapter-FaceID-Plus](https://huggingface.co/h94/IP-Adapter-FaceID/blob/main/ip-adapter-faceid-portrait_sd15.bin) for multi-reference case - place the weights to `ip_adapter/weights` directory
- [CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/h94/IP-Adapter/tree/main/models/image_encoder) - download config.json and model.safetensors, place into `ip_adapter/weights/image_encoder/` directory

You can change the directories in the config settings if you want.

## How to use

For a quick start run the following after installing all the prerequisites and downloading weights:

`python diffusion_pipeline.py --source_dir <your face photos> --target_image <your target image> --output_dir <your output directory>`

You can change the used models as well as settings via the `config.py`.

## Examples

- Single Reference:
  | Input           | Target          | Output          |
  |:---------------:|:---------------:|:---------------:|
  | <img src="assets/single_ref_input.jpg" alt="Alt text" width="300">| <img src="assets/target.jpg" alt="Alt text" width="300">|<img src="assets/single_ref.jpg" alt="Alt text" width="300">|
