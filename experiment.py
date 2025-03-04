import os
import json
import numpy as np
import torch
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Optional

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
from diffusers import StableDiffusionPipeline, DDIMScheduler



class GuidanceFunctionSegmenter:

    def __init__(self, label_map: dict = None, device: str = "cuda"):
        self.device = device
        
        # Initialise segmenter
        self.segmentator = lraspp_mobilenet_v3_large(
            LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        ).to(device)
        
        # Configs for segmenter
        self.label_map = label_map or {"cat": 8, "dog": 12}

        # Preprocessing transforms
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    
    
    def _prepare_mask(self, mask_path: str, target_size: int = 520) -> torch.Tensor:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 70
        mask = transforms.ToTensor()(mask)
        
        h, w = mask.shape[1:]
        max_size = max(h, w)
        pad_h = (max_size - h) // 2
        pad_w = (max_size - w) // 2
        
        mask = TF.pad(mask, (pad_w, pad_h, pad_w + (w % 2), pad_h + (h % 2)), fill=0)
        mask = TF.resize(mask, (target_size, target_size), interpolation=TF.InterpolationMode.BILINEAR)
        
        return mask


    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        map_img = (image + 1) * 0.5
        map_img = TF.resize(map_img, (520, 520), interpolation=TF.InterpolationMode.BILINEAR)
        map_img = self.normalize_transform(map_img)

        return map_img
    
    
    def save_guidance(self, image: torch.Tensor, output_dir, cur_class, val):
        image = self._preprocess_image(image)
        
        seg_mask = self.segmentator(
            image.to(self.device).float()
        )["out"][0, self.label_map[cur_class]].detach().cpu()
        
        # Save outputs
        seg_mask_path = os.path.join(output_dir, cur_class, f"{val}_mask.jpg")
        
        plt.imsave(seg_mask_path, seg_mask)


    def compute_loss(self, image: torch.Tensor, target: str, cur_class: str) -> torch.Tensor:
        image = self._preprocess_image(image)

        mask_path = f"./processed_data/{cur_class}/{target}_mask.jpg"
        mask = self._prepare_mask(mask_path)
        
        self.segmentator.zero_grad()
        
        seg_logits = self.segmentator(image.to(self.device).float())['out']
        class_logits = seg_logits[:, self.label_map[cur_class], :, :]
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            class_logits, 
            mask.to(self.device).expand(1, 520, 520).float()
        )
        
        return loss


class UniversalGuidance:
    def __init__(
        self,
        guidance_function,
        model_id: str = "CompVis/stable-diffusion-v1-4", 
        device: str = "cuda",
        label_map: Dict[str, int] = None
    ):
        self.device = device
        
        # Initialize Stable Diffusion pipeline
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=self.scheduler,
            revision="fp16",
            torch_dtype=torch.float16
        ).to(device)

        # Initialize segmentation model
        self.guidance_function = guidance_function

    
    def _create_custom_noise_schedule(self) -> np.ndarray:
        def custom_mapping(x: int) -> float:
            x = np.clip(x, 1, 999)
            if x <= 550:
                return (-(x + 500) * (x - 550) / 1200 + 400) - 40
            elif x <= 950:
                return 400 * np.exp(-2 * (x - 550) / 400) - 40
            else:
                return 400 * np.exp(-4) * (1 - 0.1 * (x - 900) / 99) - 4

        x_values = np.arange(0, 1000)
        y_values = np.array([custom_mapping(x) for x in x_values])
        return np.flip(y_values)


    def generate_and_segment(
        self, 
        annotations_path: str, 
        output_dir: str, 
        cur_class: str = "cat", 
        num_inference_steps: int = 500,
        guidance_scale: float = 1.5
    ):
        # Negative prompt to discourage undesirable image characteristics
        negative_prompt = "blurry, out of focus, low resolution, pixelated, distorted, unnatural colors, extra limbs, deformed face, unrealistic proportions, overly smooth, oversaturated, artifacts, grainy, noisy, overexposed, underexposed, bad lighting, unnatural textures, painting, cartoon, anime, CGI, 3D render, watermark, text, cropped, cut off, unnatural pose, uncanny valley, mutated, exaggerated features, artificial details, unrealistic fur"

        s_schedule = self._create_custom_noise_schedule()

        # Load annotations
        with open(annotations_path, "r") as f:
            annotations = json.load(f)

        # Ensure output directory exists
        os.makedirs(os.path.join(output_dir, cur_class), exist_ok=True)

        for val, key in annotations.items():
            # Encoding prompt
            with torch.no_grad():
                prompt_embeds = self.pipe._encode_prompt(
                    key,
                    self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    lora_scale=None,
                )

            # Prepare timesteps and latents
            self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.pipe.scheduler.timesteps
            latents = self._initialize_latents(prompt_embeds)
            latents.requires_grad_(False)

            # Denoising loop
            latents = self._denoising_loop(
                latents, 
                prompt_embeds, 
                timesteps,
                s_schedule, 
                guidance_scale,
                cur_class,
                val
            )

            # Decode and save results
            self._save_results(latents, val, cur_class, output_dir)


    def _initialize_latents(self, prompt_embeds):
        num_images = 1
        num_channels_latents = self.pipe.unet.config.in_channels
        height = self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        width = height

        return self.pipe.prepare_latents(
            num_images,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator=None,
            latents=None,
        )

    def _denoising_loop(
        self, 
        latents, 
        prompt_embeds, 
        timesteps,
        s_schedule, 
        guidance_scale,
        cur_class,
        val,
        k: int = 10
    ):
        for i, t in tqdm(enumerate(timesteps), total=500):
            for _ in range(k):
                z_t = latents.detach().clone()
                z_t.requires_grad_(True)
                
                # Classifier-free guidance
                latent_model_input = z_t.repeat(2, 1, 1, 1)
                latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

                # Zero gradients
                self.pipe.unet.zero_grad()

                # Predict noise
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]

                # Classifier-free guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute alphas
                alpha_prod_t = self.pipe.scheduler.alphas_cumprod[timesteps[i]]
                alpha_prod_t_prev = (
                    self.pipe.scheduler.alphas_cumprod[timesteps[i + 1]] 
                    if i + 1 < len(timesteps) 
                    else torch.tensor(1.0)
                )

                self.pipe.vae.zero_grad()

                # Denoise
                z_zero = (z_t - (1 - alpha_prod_t) ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
                img = self.pipe.vae.decode(z_zero / self.pipe.vae.config.scaling_factor, return_dict=False)[0]

                # Guidance loss
                loss = self.guidance_function.compute_loss(img, val, cur_class)

                loss.backward()
                noise_pred += s_schedule[t] * z_t.grad

                next_latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
                latents = (alpha_prod_t/alpha_prod_t_prev)**0.5 * next_latents \
                    + (1 - alpha_prod_t/alpha_prod_t_prev)**0.5 * torch.randn_like(next_latents)

            latents = next_latents

        return latents

    def _save_results(self, latents, val, cur_class, output_dir):
        with torch.no_grad():
            image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]

            # Save guidance
            self.guidance_function.save_guidance(image, output_dir, cur_class, val)

            # Save outputs
            output_path = os.path.join(output_dir, cur_class, f"{val}.jpg")
            
            generated_image = self.pipe.image_processor.postprocess(
                image.detach(), 
                output_type="pil", 
                do_denormalize=[True]
            )[0]
            
            generated_image.save(output_path)


def main():
    guidance_function = GuidanceFunctionSegmenter()
    
    generator = UniversalGuidance(guidance_function)
    generator.generate_and_segment(
        annotations_path="./processed_data/cat/annotations.json",
        output_dir="./forward_pass",
        cur_class="cat"
    )


if __name__ == "__main__":
    main()