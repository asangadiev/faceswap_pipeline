import os


src_dir = os.getcwd()

class Config:
    def __init__(self): 
        # Model settings
        self.face_analyzer = "buffalo_l"
        self.face_parser = "farl/celebm/448"
        self.base_model = "SG161222/Realistic_Vision_V4.0_noVAE"
        self.vae_model = "stabilityai/sd-vae-ft-mse"
        self.image_encoder_path = os.path.join(src_dir, "ip_adapter", "weights", "image_encoder")
        self.ip_portrait_path = os.path.join(src_dir, "ip_adapter", "weights", "ip-adapter-faceid-portrait_sd15.bin")
        self.ip_plus_path = os.path.join(src_dir, "ip_adapter", "weights", "ip-adapter-faceid-plus_sd15.bin")
        self.device = "cuda"

        # Diffusion settings
        self.prompt = "high quality photo, highly detailed face, best quality"
        self.negative_prompt = "blurry, low quality, distorted, deformed, lowres, bad anatomy, monochrome"
        self.noise_scheduler = "ddim"
        self.num_inference_steps = 100
        self.guidance_scale = 7.5
        self.strength = 0.6
        self.diffusion_seed = 42

        # GFPGAN settings
        self.face_restoration = True
        self.gfpgan_model = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        self.gfpgan_upscale = 2
        self.gfpgan_arch = "clean"
        self.gfpgan_bg_upsampler = None

        # General settings
        self.source_dir = os.path.join(src_dir, "data", "reference")
        self.target_image = os.path.join(src_dir, "data", "target", "IMG_9236.jpg")
        self.output_dir = os.path.join(src_dir, "data", "output")

        # Segmentation mask settings (0 - disabled, 1 - enabled)
        self.active_regions = [
            0, # background
            1, # "neck"
            1, # "face"
            0, # "cloth"
            1, # "rr"
            1, # "lr"
            1, # "rb"
            1, # "lb"
            1, # "re"
            1, # "le"
            1, # "nose"
            1, # "imouth"
            1, # "llip"
            1, # "ulip"
            0, # "hair"
            1, # "eyeg"
            0, # "hat"
            0, # "earr"
            0, # "neck_l"
        ]