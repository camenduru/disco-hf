import gradio as gr
import os, requests
from inference import setup_model, colorize_grayscale, predict_anchors

## download checkpoint
def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)

id = "1J4vB6kG4xBLUUKpXr5IhnSSa4maXgRvQ"
destination = "disco-beta.pth.rar"
download_file_from_google_drive(id, destination)
os.rename("disco-beta.pth.tar", "./checkpoints/disco-beta.pth.tar")

## step 1: set up model
device = "cuda"
checkpt_path = "./checkpoints/disco-beta.pth.tar"
assert os.path.exists(checkpt_path), "No checkpoint found!"
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

demo = gr.Blocks(title="DISCO: Image Colorization")
with demo:
    gr.Markdown(value="""**DISCO: image colorization that disentangles color multimodality and spatial affinity via global anchors**.""")
    with gr.Row():
        with gr.Column(scale=1):
            Image_input = gr.Image(type="numpy", label="Input", interactive=True)
            Image_anchor = gr.Image(type="numpy", label="Anchor", tool="color-sketch", interactive=True, visible=False)
            with gr.Row():
                Num_anchor = gr.Number(type="int", value=8, label="Num. of anchors (3~14)")
                Radio_resolution = gr.Radio(type="index", choices=["Low (256x256)", "High (512x512)"], \
                                            label="Colorization resolution", value="Low (256x256)")
            Ckeckbox_editable = gr.Checkbox(default=False, label='Show editable anchors')
            with gr.Row():
                Button_show_anchor = gr.Button(value="Predict anchors", visible=False)
                Button_run = gr.Button(value="Colorize")
        with gr.Column(scale=1):
            Image_output = gr.Image(type="numpy", label="Output", shape=[100,100])
    
    Ckeckbox_editable.change(fn=switch_states, inputs=Ckeckbox_editable, outputs=[Image_anchor, Button_show_anchor])
    Button_show_anchor.click(fn=click_predanchors, inputs=[Image_input, Num_anchor, Radio_resolution, Ckeckbox_editable], outputs=Image_anchor)
    Button_run.click(fn=click_colorize, inputs=[Image_input, Image_anchor, Num_anchor, Radio_resolution, Ckeckbox_editable], \
                    outputs=Image_output)
    ## guiline
    gr.Markdown(value="""
                    **Guideline**
                    1. Upload your image;
                    2. Set up the arguments: "Num. of anchors" and "Colorization resolution";
                    3. Two modes are supported:
                        - **Editable**: check ""Show editable anchors" and click "Predict anchors". Then, modify the colors of the predicted anchors (anchor mask will be applied afterward). Finally, click "Colorize" to get the result.
                        - **Automatic**: click "Colorize" to get the automatically colorized output.
                    
                    *To know more about the method, please refer to our project page: [https://menghanxia.github.io/projects/disco.html](https://menghanxia.github.io/projects/disco.html)*
                    """)
    
demo.launch(server_name='9.134.253.83',server_port=7788)