import os
import argparse

import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipelineLegacy, AutoencoderKL, DPMSolverMultistepScheduler, DDIMScheduler
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID

from config import Config
from face_pipeline import FacePipeline


def run_pipeline(config):
    face_pipeline = FacePipeline(config)
    input_images = os.listdir(config.source_dir)
    n_references = len(input_images)

    face_embeddings = []
    for input_image in input_images:
        source_image = face_pipeline.get_source_face(os.path.join(config.source_dir, input_image))
        face_embeddings.append(source_image['face_embeddings'].unsqueeze(0))

    face_embeddings = torch.cat(face_embeddings, dim=1)

    target_image = face_pipeline.get_target_image(config.target_image)

    if config.noise_scheduler == "ddim":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
    elif config.noise_scheduler == "dpmsolver_karras":
        noise_scheduler = DPMSolverMultistepScheduler(
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            num_train_timesteps=1000,
            steps_offset=1,
        )

    vae = AutoencoderKL.from_pretrained(config.vae_model).half()

    pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )

    ip_model = IPAdapterFaceID(pipe, config.ip_ckpt, config.device, num_tokens=16, n_cond=n_references)

    images = ip_model.generate(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        faceid_embeds=face_embeddings,
        image=target_image['face_crop'],
        mask_image=target_image['face_mask'],
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        strength=config.strength,
        seed=config.diffusion_seed,
        num_samples=1
    )

    images = [face_pipeline.project_face(target_image, image) for image in images]

    del pipe, ip_model
    torch.cuda.empty_cache()
    if config.face_restoration:
        orig_sizes = [image.size for image in images]
        images = [face_pipeline.enhance_face(image).resize(orig_size, resample=Image.Resampling.LANCZOS) for image, orig_size in zip(images, orig_sizes)]

    return images


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--source_dir", type=str, default="data/reference")
    args.add_argument("--target_image", type=str, default="data/target/IMG_9236.jpg")
    args.add_argument("--output_dir", type=str, default="data/output")
    args.add_argument("--output_name", type=str, default="output")
    args = args.parse_args()

    config = Config()
    config.source_dir = args.source_dir
    config.output_dir = args.output_dir
    config.target_image = args.target_image

    images = run_pipeline(config)

    for i, image in enumerate(images):
        image.save(os.path.join(config.output_dir, f"{args.output_name}_{i}.png"))