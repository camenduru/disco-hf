import gradio as gr
import os, requests
import numpy as np
from inference import setup_model, colorize_grayscale, predict_anchors

## local |  remote
RUN_MODE = "local"
if RUN_MODE != "local":
    os.system("wget https://huggingface.co/menghanxia/disco/resolve/main/disco-beta.pth.rar")
    os.rename("disco-beta.pth.rar", "./checkpoints/disco-beta.pth.rar")

## step 1: set up model
device = "cpu"
checkpt_path = "checkpoints/disco-beta.pth.rar"
colorizer, colorLabeler = setup_model(checkpt_path, device=device)

def click_colorize(rgb_img, hint_img, n_anchors, is_high_res, is_editable):
    if hint_img is None:
        hint_img = rgb_img
    output = colorize_grayscale(colorizer, colorLabeler, rgb_img, hint_img, n_anchors, is_high_res, is_editable, device)
    return output

def click_predanchors(rgb_img, n_anchors, is_high_res, is_editable):
    output = predict_anchors(colorizer, colorLabeler, rgb_img, n_anchors, is_high_res, is_editable, device)
    return output

## step 2: configure interface
def switch_states(is_checked):
    if is_checked:
        return gr.Image.update(visible=True), gr.Button.update(visible=True)
    else:
        return gr.Image.update(visible=False), gr.Button.update(visible=False)

demo = gr.Blocks(title="DISCO")
with demo:
    gr.Markdown(value="""
                    **Gradio demo for DISCO: Disentangled Image Colorization via Global Anchors**. Check our project page [*Here*](https://menghanxia.github.io/projects/disco.html).
                    """)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                Image_input = gr.Image(type="numpy", label="Input", interactive=True)
                Image_anchor = gr.Image(type="numpy", label="Anchor", tool="color-sketch", interactive=True, visible=False)
            with gr.Row():
                Num_anchor = gr.Number(type="int", value=8, label="Num. of anchors (3~14)")
                Radio_resolution = gr.Radio(type="index", choices=["Low (256x256)", "High (512x512)"], \
                                                label="Colorization resolution (Low is more stable)", value="Low (256x256)")
            with gr.Row():
                Ckeckbox_editable = gr.Checkbox(default=False, label='Show editable anchors')
                Button_show_anchor = gr.Button(value="Predict anchors", visible=False)
            Button_run = gr.Button(value="Colorize")
        with gr.Column():
            Image_output = gr.Image(type="numpy", label="Output").style(height=480)

    Ckeckbox_editable.change(fn=switch_states, inputs=Ckeckbox_editable, outputs=[Image_anchor, Button_show_anchor])
    Button_show_anchor.click(fn=click_predanchors, inputs=[Image_input, Num_anchor, Radio_resolution, Ckeckbox_editable], outputs=Image_anchor)
    Button_run.click(fn=click_colorize, inputs=[Image_input, Image_anchor, Num_anchor, Radio_resolution, Ckeckbox_editable], \
                    outputs=Image_output)
    ## guiline
    gr.Markdown(value="""    
                    **Guideline**
                    1. upload your image; &nbsp; [*example images*](https://1drv.ms/u/s!Al07CQ4AbykHkmk2dFpNPA8VObk4?e=1C7D3R)😛
                    2. set up the arguments: "Num. of anchors" and "Colorization resolution";
                    3. run the colorization (two modes supported):
                        - *Automatic mode*: click "Colorize" to get the automatically colorized output.
                        - *Editable mode*: check ""Show editable anchors" and click "Predict anchors". Then, modify the colors of the predicted anchors (only anchor region will be used). Finally, click "Colorize" to get the result.
                    """)
    gr.HTML(value="""
                <p style="text-align:center; color:orange"><a href='https://menghanxia.github.io/projects/disco.html' target='_blank'>DISCO Project Page</a> | <a href='https://github.com/MenghanXia/DisentangledColorization' target='_blank'>Github Repo</a></p>
                    """)

if RUN_MODE == "local":
    demo.launch(server_name='9.134.253.83',server_port=7788)
else:
    demo.launch()