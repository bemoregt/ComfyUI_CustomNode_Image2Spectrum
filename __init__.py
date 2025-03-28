import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy.fft import fft2, fftshift

class Image_Spectrum:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "amplitude": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 50,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_spectrum"
    CATEGORY = "image"

    def image_spectrum(self, image, amplitude):
        image = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        image = ImageOps.grayscale(image)
        
        # 2D Fourier 변환 수행
        f_transform = fft2(image)
        f_shift = fftshift(f_transform)
        spectrum = amplitude*np.log1p(np.abs(f_shift))
        
        image = np.array(spectrum).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(image)
        image = torch.from_numpy(image)[None,]
        
        return (image,)

NODE_CLASS_MAPPINGS = {
    "Image_Spectrum": Image_Spectrum
}
