import os
from cog import BasePredictor, Input, Path

import os, requests
import numpy as np
from inference import setup_model, colorize_grayscale, predict_anchors
    
def inference(rgb_img, n_anchors, is_high_res, is_editable, colorizer, colorLabeler):
    hint_img = rgb_img
    output = colorize_grayscale(colorizer, colorLabeler, rgb_img, hint_img, n_anchors, is_high_res, is_editable, "cpu")
    return output

class Predictor(BasePredictor):
    def setup(self) -> None:
        os.system("wget https://huggingface.co/menghanxia/disco/resolve/main/disco-beta.pth.rar -O checkpoints/disco-beta.pth.rar")
        device = "cpu"
        checkpt_path = "checkpoints/disco-beta.pth.rar"
        self.colorizer, self.colorLabeler = setup_model(checkpt_path, device="cpu")
    def predict(
        self,
        rgb_img: Path = Input(description="Input Image"),
    ) -> Path:
        output_image = inference(rgb_img, 8, "High (512x512)", False, self.colorizer, self.colorLabeler)
        return Path(output_image)