import os
from cog import BasePredictor, Input, Path

import os, requests
import numpy as np
from inference import setup_model, colorize_grayscale, predict_anchors
    
def inference(rgb_img, hint_img, n_anchors, is_high_res, is_editable, colorizer, colorLabeler):
    if hint_img is None:
        hint_img = rgb_img
    output = colorize_grayscale(colorizer, colorLabeler, rgb_img, hint_img, n_anchors, is_high_res, is_editable, device)
    return output

class Predictor(BasePredictor):
    def setup(self) -> None:
        device = "cpu"
        checkpt_path = "checkpoints/disco-beta.pth.rar"
        self.colorizer, self.colorLabeler = setup_model(checkpt_path, device=device)
    def predict(
        self,
        input_image: Path = Input(description="Input Image"),
    ) -> Path:
        output_image = inference(rgb_img, hint_img, n_anchors, is_high_res, is_editable, self.colorizer, self.colorLabeler)
        return Path(output_image)