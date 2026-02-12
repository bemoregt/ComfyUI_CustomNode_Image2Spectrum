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

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("amplitude_spectrum", "phase_spectrum")
    FUNCTION = "image_spectrum"
    CATEGORY = "image"

    def image_spectrum(self, image, amplitude):
        image = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        image = ImageOps.grayscale(image)

        # 2D Fourier 변환 수행
        f_transform = fft2(image)
        f_shift = fftshift(f_transform)

        # 진폭 스펙트럼
        spectrum = amplitude * np.log1p(np.abs(f_shift))
        amp_img = np.array(spectrum).astype(np.float32) / 255.0
        amp_tensor = torch.from_numpy(amp_img)[None,]

        # 위상 스펙트럼 (-π ~ π → 0 ~ 1 정규화)
        phase = np.angle(f_shift)
        phase_img = ((phase + np.pi) / (2 * np.pi)).astype(np.float32)
        phase_tensor = torch.from_numpy(phase_img)[None,]

        return (amp_tensor, phase_tensor)

NODE_CLASS_MAPPINGS = {
    "Image_Spectrum": Image_Spectrum
}
