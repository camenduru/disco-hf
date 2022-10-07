# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import tempfile
import os, glob
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import model, basic
from utils import util

class Predictor(BasePredictor):
    def setup(self):
        seed = 130
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        #print('--------------', torch.cuda.is_available())
        """Load the model into memory to make running multiple predictions efficient"""
        self.colorizer = model.AnchorColorProb(inChannel=1, outChannel=313, enhanced=True)
        self.colorizer = self.colorizer.cuda()
        checkpt_path = "./checkpoints/disco-beta.pth.rar"
        assert os.path.exists(checkpt_path)
        data_dict = torch.load(checkpt_path, map_location=torch.device('cpu'))
        self.colorizer.load_state_dict(data_dict['state_dict'])
        self.colorizer.eval()
        self.color_class = basic.ColorLabel(lambda_=0.5, device='cuda')
    
    def resize_ab2l(self, gray_img, lab_imgs):
        H, W = gray_img.shape[:2]
        reszied_ab = cv2.resize(lab_imgs[:,:,1:], (W,H), interpolation=cv2.INTER_LINEAR)
        return np.concatenate((gray_img, reszied_ab), axis=2)
    
    def predict(
        self,
        image: Path = Input(description="input image. Output will be one or multiple colorized images."),
        n_anchors: int = Input(
            description="number of color anchors", ge=3, le=14, default=8
        ),
        multi_result: bool = Input(
            description="to generate diverse results", default=False
        ),
        vis_anchors: bool = Input(
            description="to visualize the anchor locations", default=False
        )
    ) -> Path:
        """Run a single prediction on the model"""
        bgr_img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_img = np.array(rgb_img / 255., np.float32)
        lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        org_grays = (lab_img[:,:,[0]]-50.) / 50.
        lab_img = cv2.resize(lab_img, (256,256), interpolation=cv2.INTER_LINEAR)
        
        lab_img = torch.from_numpy(lab_img.transpose((2, 0, 1)))
        gray_img = (lab_img[0:1,:,:]-50.) / 50.
        ab_chans = lab_img[1:3,:,:] / 110.
        input_grays = gray_img.unsqueeze(0)
        input_colors = ab_chans.unsqueeze(0)
        input_grays = input_grays.cuda(non_blocking=True)
        input_colors = input_colors.cuda(non_blocking=True)
                
        sampled_T = 2 if multi_result else 0
        pal_logit, ref_logit, enhanced_ab, affinity_map, spix_colors, hint_mask = self.colorizer(input_grays, \
                                                            input_colors, n_anchors, True, sampled_T)
        pred_probs = pal_logit
        guided_colors = self.color_class.decode_ind2ab(ref_logit, T=0)
        sp_size = 16
        guided_colors = basic.upfeat(guided_colors, affinity_map, sp_size, sp_size)
        res_list = []
        if multi_result:
            for no in range(3):
                pred_labs = torch.cat((input_grays,enhanced_ab[no:no+1,:,:,:]), dim=1)
                lab_imgs = basic.tensor2array(pred_labs).squeeze(axis=0)
                lab_imgs = self.resize_ab2l(org_grays, lab_imgs)
                #util.save_normLabs_from_batch(lab_imgs, save_dir, [file_name], -1, suffix='c%d'%no)
                res_list.append(lab_imgs)
        else:
            pred_labs = torch.cat((input_grays,enhanced_ab), dim=1)
            lab_imgs = basic.tensor2array(pred_labs).squeeze(axis=0)
            lab_imgs = self.resize_ab2l(org_grays, lab_imgs)
            #util.save_normLabs_from_batch(lab_imgs, save_dir, [file_name], -1)#, suffix='enhanced')
            res_list.append(lab_imgs)
        
        if vis_anchors:
            ## visualize anchor locations
            anchor_masks = basic.upfeat(hint_mask, affinity_map, sp_size, sp_size)
            marked_labs = basic.mark_color_hints(input_grays, enhanced_ab, anchor_masks, base_ABs=enhanced_ab)
            hint_imgs = basic.tensor2array(marked_labs).squeeze(axis=0)
            hint_imgs = self.resize_ab2l(org_grays, hint_imgs)
            #util.save_normLabs_from_batch(hint_imgs, save_dir, [file_name], -1, suffix='anchors')
            res_list.append(hint_imgs)
        
        output = cv2.vconcat(res_list)
        output[:,:,0] = output[:,:,0] * 50.0 + 50.0
        output[:,:,1:3] = output[:,:,1:3] * 110.0
        rgb_output = cv2.cvtColor(output[:,:,:], cv2.COLOR_LAB2BGR)      
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        cv2.imwrite(str(out_path), (rgb_output*255.0).astype(np.uint8))
        return out_path
