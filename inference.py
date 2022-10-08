import os, glob, sys, logging
import argparse, datetime, time
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import model, basic
from utils import util


def setup_model(checkpt_path, device="cuda"):
    #print('--------------', torch.cuda.is_available())
    """Load the model into memory to make running multiple predictions efficient"""
    colorLabeler = basic.ColorLabel(device=device)
    colorizer = model.AnchorColorProb(inChannel=1, outChannel=313, enhanced=True, colorLabeler=colorLabeler)
    colorizer = colorizer.to(device)
    #checkpt_path = "./checkpoints/disco-beta.pth.rar"
    assert os.path.exists(checkpt_path), "No checkpoint found!"
    data_dict = torch.load(checkpt_path, map_location=torch.device('cpu'))
    colorizer.load_state_dict(data_dict['state_dict'])
    colorizer.eval()
    return colorizer, colorLabeler


def resize_ab2l(gray_img, lab_imgs, vis=False):
    H, W = gray_img.shape[:2]
    reszied_ab = cv2.resize(lab_imgs[:,:,1:], (W,H), interpolation=cv2.INTER_LINEAR)
    if vis:
        gray_img = cv2.resize(lab_imgs[:,:,:1], (W,H), interpolation=cv2.INTER_LINEAR)
        return np.concatenate((gray_img[:,:,np.newaxis], reszied_ab), axis=2)
    else:
        return np.concatenate((gray_img, reszied_ab), axis=2)

def prepare_data(rgb_img, target_res):
    rgb_img = np.array(rgb_img / 255., np.float32)
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    org_grays = (lab_img[:,:,[0]]-50.) / 50.
    lab_img = cv2.resize(lab_img, target_res, interpolation=cv2.INTER_LINEAR)
        
    lab_img = torch.from_numpy(lab_img.transpose((2, 0, 1)))
    gray_img = (lab_img[0:1,:,:]-50.) / 50.
    ab_chans = lab_img[1:3,:,:] / 110.
    input_grays = gray_img.unsqueeze(0)
    input_colors = ab_chans.unsqueeze(0)
    return input_grays, input_colors, org_grays


def colorize_grayscale(colorizer, color_class, rgb_img, hint_img, n_anchors, is_high_res, is_editable, device="cuda"):
    n_anchors = int(n_anchors)
    n_anchors = max(n_anchors, 3)
    n_anchors = min(n_anchors, 14)
    target_res = (512,512) if is_high_res else (256,256)
    input_grays, input_colors, org_grays = prepare_data(rgb_img, target_res)
    input_grays = input_grays.to(device)
    input_colors = input_colors.to(device)
    
    if is_editable:
        print('>>>:editable mode')
        sampled_T = -1
        _, input_colors, _ = prepare_data(hint_img, target_res)
        input_colors = input_colors.to(device)
        pal_logit, ref_logit, enhanced_ab, affinity_map, spix_colors, hint_mask = colorizer(input_grays, \
                                                                input_colors, n_anchors, sampled_T)
    else:
        print('>>>:automatic mode')
        sampled_T = 0
        pal_logit, ref_logit, enhanced_ab, affinity_map, spix_colors, hint_mask = colorizer(input_grays, \
                                                                input_colors, n_anchors, sampled_T)

    pred_labs = torch.cat((input_grays,enhanced_ab), dim=1)
    lab_imgs = basic.tensor2array(pred_labs).squeeze(axis=0)
    lab_imgs = resize_ab2l(org_grays, lab_imgs)
        
    lab_imgs[:,:,0] = lab_imgs[:,:,0] * 50.0 + 50.0
    lab_imgs[:,:,1:3] = lab_imgs[:,:,1:3] * 110.0
    rgb_output = cv2.cvtColor(lab_imgs[:,:,:], cv2.COLOR_LAB2RGB)
    return (rgb_output*255.0).astype(np.uint8)


def predict_anchors(colorizer, color_class, rgb_img, n_anchors, is_high_res, is_editable, device="cuda"):
    n_anchors = int(n_anchors)
    n_anchors = max(n_anchors, 3)
    n_anchors = min(n_anchors, 14)
    target_res = (512,512) if is_high_res else (256,256)
    input_grays, input_colors, org_grays = prepare_data(rgb_img, target_res)
    input_grays = input_grays.to(device)
    input_colors = input_colors.to(device)
                
    sampled_T, sp_size = 0, 16
    pal_logit, ref_logit, enhanced_ab, affinity_map, spix_colors, hint_mask = colorizer(input_grays, \
                                                            input_colors, n_anchors, sampled_T)
    pred_probs = pal_logit
    guided_colors = color_class.decode_ind2ab(ref_logit, T=0)
    guided_colors = basic.upfeat(guided_colors, affinity_map, sp_size, sp_size)
    anchor_masks = basic.upfeat(hint_mask, affinity_map, sp_size, sp_size)
    marked_labs = basic.mark_color_hints(input_grays, guided_colors, anchor_masks, base_ABs=None)
    lab_imgs = basic.tensor2array(marked_labs).squeeze(axis=0)
    lab_imgs = resize_ab2l(org_grays, lab_imgs, vis=True)
        
    lab_imgs[:,:,0] = lab_imgs[:,:,0] * 50.0 + 50.0
    lab_imgs[:,:,1:3] = lab_imgs[:,:,1:3] * 110.0
    rgb_output = cv2.cvtColor(lab_imgs[:,:,:], cv2.COLOR_LAB2RGB)
    return (rgb_output*255.0).astype(np.uint8)