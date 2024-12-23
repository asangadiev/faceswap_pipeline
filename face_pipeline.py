import cv2
import facer
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from facer import face_parser
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from facexlib.utils.face_restoration_helper import FaceRestoreHelper


class FacePipeline:
    def __init__(self, config):
        self.config = config

        self.face_analyzer = FaceAnalysis(name=config.face_analyzer, providers=['CPUExecutionProvider'])
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

        self.face_parser = face_parser(config.face_parser,
                                       "cpu",
                                       model_path="https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.celebm.main_ema_181500_jit.pt",
                                       )
        
        if config.face_restoration:
            if config.gfpgan_bg_upsampler == "realesrgan":
                bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
                    tile=400,
                    tile_pad=10,
                    pre_pad=0,
                    half=True)
            else:
                bg_upsampler = None
            
            self.face_restorer = GFPGANer(
                model_path=config.gfpgan_model,
                upscale=config.gfpgan_upscale,
                arch=config.gfpgan_arch,
                device=config.device,
                bg_upsampler=bg_upsampler,
                channel_multiplier=2
            )

            self.face_restorer.face_helper = FaceRestoreHelper(
                config.gfpgan_upscale,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=False,
                device=config.device,
                model_rootpath='gfpgan/weights')

    def parse_face(self, image: np.ndarray) -> np.ndarray:
        image = facer.hwc2bchw(torch.from_numpy(image))
        with torch.inference_mode():
            logits, _ = self.face_parser.net(image / 255.0)
            mask = logits.argmax(dim=1)
        return mask
    
    def load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def get_source_face(self, image_path: str) -> np.ndarray:
        source_image = {}
        
        image = self.load_image(image_path)

        face_annotations = self.face_analyzer.get(image)[0]
        source_image['face'] = face_align.norm_crop(image, landmark=face_annotations.kps, image_size=224)
        source_image['face_embeddings'] = torch.from_numpy(face_annotations.normed_embedding).unsqueeze(0)

        return source_image
    
    def get_target_image(self, image_path: str) -> np.ndarray:
        target_image = {}

        image = self.load_image(image_path)
        target_image['image'] = Image.fromarray(image)

        # Generate a face mask
        face_annotations = self.face_analyzer.get(image)[0]
        bbox = face_annotations.bbox.astype(int)
        face_crop = self.crop_image(image, bbox)

        face_regions = torch.tensor([i for i, region in enumerate(self.config.active_regions) if region == 1])
        face_mask = self.parse_face(face_crop)
        face_mask = torch.isin(face_mask, face_regions).float()
        face_mask = F.interpolate(face_mask.unsqueeze(0), size=(bbox[3]-bbox[1], bbox[2]-bbox[0]), mode='nearest')[0]

        image_mask = torch.zeros((1, image.shape[0], image.shape[1]), dtype=face_mask.dtype, device=face_mask.device)
        image_mask[0, bbox[1]:bbox[3], bbox[0]:bbox[2]] = face_mask[0]
        image_mask = image_mask.permute(1, 2, 0).numpy()

        target_image['image_mask'] = Image.fromarray((image_mask[..., 0] * 255).clip(0, 255).astype(np.uint8))

        # Expand bbox to square while maintaining center
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        max_size = max(bbox_width, bbox_height)
        
        # Calculate required padding
        pad_width = (max_size - bbox_width) // 2
        pad_height = (max_size - bbox_height) // 2
        
        # Adjust bbox while ensuring it stays within image bounds
        new_bbox = np.array([
            max(0, bbox[0] - pad_width),
            max(0, bbox[1] - pad_height),
            min(image.shape[1], bbox[2] + pad_width),
            min(image.shape[0], bbox[3] + pad_height)
        ])
        target_image['new_bbox_coords'] = new_bbox
        
        # Get square crop using new bbox
        target_image['face_crop'] = Image.fromarray(self.crop_image(image, new_bbox)).resize((512, 512), resample=Image.Resampling.LANCZOS)

        # Get new square mask
        face_mask = F.interpolate(face_mask.unsqueeze(0), size=(new_bbox[3]-new_bbox[1], new_bbox[2]-new_bbox[0]), mode='nearest')[0]
        face_mask = face_mask.permute(1, 2, 0).numpy().astype(np.uint8)
        target_image['face_mask'] = Image.fromarray((face_mask[..., 0] * 255).astype(np.uint8)).resize((512, 512), resample=Image.Resampling.NEAREST)
        
        return target_image
    
    def project_face(self, target_image: dict, inpainted_face: Image.Image) -> Image.Image:
        # Convert images to numpy arrays
        original = np.array(target_image['image'])
        #face = np.array(inpainted_face)
        mask = np.array(target_image['image_mask'])
        
        # Get the bounding box dimensions from target_image
        new_bbox = target_image['new_bbox_coords']
        
        # Resize inpainted face back to bbox size
        face = np.array(inpainted_face.resize(
            (new_bbox[2] - new_bbox[0], new_bbox[3] - new_bbox[1]), 
            resample=Image.Resampling.LANCZOS
        ))
        
        # Ensure mask is in correct format (H,W,1)
        if len(mask.shape) == 2:
            mask = mask[..., None]
        
        # Normalize mask to [0,1]
        mask = mask.astype(float) / 255.0
        # Apply Gaussian blur to the mask
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        # Ensure values stay in [0,1] range
        mask = mask.clip(0, 1)[..., None]
        
        # Create output image
        result = original.copy()
        
        # Paste face into position using mask
        result[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]] = (
            result[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]] * (1 - mask[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]) + 
            face * mask[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]
        )
        
        # Convert back to uint8
        result = result.clip(0, 255).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def enhance_face(self, image: Image) -> np.ndarray:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        _, _, restored_img = self.face_restorer.enhance(image,
                                                        has_aligned=False,
                                                        only_center_face=False,
                                                        paste_back=True,
                                                        weight=0.5)
        return Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))

    def image_grid(self, images: list[np.ndarray], rows: int, cols: int) -> None:
        _, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 6))
    
        # Convert axes to array if single subplot
        if rows * cols == 1:
            axes = np.array([axes])
        
        for ax, image in zip(axes.flatten(), images):
            ax.imshow(image)
            ax.axis('off')
        
        plt.show()

    def crop_image(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x_min, y_min, x_max, y_max = bbox.astype(int)
        return image[y_min:y_max, x_min:x_max]