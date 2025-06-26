import gradio as gr
from torchvision.transforms.functional import to_tensor
from huggingface_hub import hf_hub_download

from t2i_inference import T2IConfig, TextToImageInference

def generate_text(self, image: str, prompt: str) -> str:
    image = image.convert('RGB')
    image = to_tensor(image).unsqueeze(0).to(self.device)
    
    image_code = self.visual_tokenizer.encoder(image.to(self.config.dtype))['bottleneck_rep']
    image_text = "".join([f"<I{x}>" for x in image_code[0].cpu().tolist()])
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{image_text}\n{prompt}"}
    ]
    
    input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = self.tokenizer(input_text, return_tensors="pt")
    
    gen_ids = self.model.generate(
        inputs.input_ids.to(self.device),
        max_new_tokens=512,
        do_sample=True)
    return self.tokenizer.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

config = T2IConfig()
config.ar_path = hf_hub_download("csuhan/TA-Tok", "ar_dtok_lp_512px.pth")
config.encoder_path = hf_hub_download("csuhan/TA-Tok", "ta_tok.pth")
config.decoder_path = hf_hub_download("peizesun/llamagen_t2i", "vq_ds16_t2i.pt")
inference = TextToImageInference(config)

def generate_image(prompt, top_p, top_k, cfg_scale):
    config.top_p = top_p
    config.top_k = top_k
    config.cfg_scale = cfg_scale
    image = inference.generate_image(prompt)
    return image

def clear_inputs_t2i():
    return "", None

def understand_image(image, prompt):
    return generate_text(inference, image, prompt)

def clear_inputs_i2t():
    return None, ""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div align="center">

        ### Tar: Unifying Visual Understanding and Generation via Text-Aligned Representations  

        [🕸️ Project Page](http://tar.csuhan.com) • [📄 Paper](http://arxiv.org/abs/2506.18898) • [💻 Code](https://github.com/csuhan/Tar) • [📦 Model](https://huggingface.co/csuhan/TA-Tok)

        </div>
        """,
        elem_id="title",
    )
    with gr.Tab("Image Generation"):
      with gr.Row():
          with gr.Column(scale=1):
              prompt = gr.Textbox(label="Prompt", placeholder="Enter a prompt")
              with gr.Accordion("Advanced Settings", open=False):
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
                top_k = gr.Slider(1, 2000, value=1200, step=10, label="Top-k")
                cfg_scale = gr.Slider(1.0, 20.0, value=4.0, step=0.5, label="CFG Scale")
              with gr.Row():
                  generate_btn = gr.Button("Generate")
                  clear_btn = gr.Button("Clear")
          with gr.Column(scale=2):
              output_image = gr.Image(label="Generated Image")

      generate_btn.click(
          generate_image, 
          inputs=[prompt, top_p, top_k, cfg_scale], 
          outputs=output_image
      )
      clear_btn.click(
          clear_inputs_t2i, 
          outputs=[prompt, output_image]
      )

    with gr.Tab("Image Understanding"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Image", type="pil")
                question_input = gr.Textbox(label="Instruction", value="Describe the image shortly.")
                with gr.Row():
                    qa_btn = gr.Button("Generate")
                    clear_btn_i2t = gr.Button("Clear")
            with gr.Column(scale=1):
                answer_output = gr.Textbox(label="Response", lines=4)

        qa_btn.click(
            understand_image,
            inputs=[image_input, question_input],
            outputs=answer_output
        )

        clear_btn_i2t.click(
            clear_inputs_i2t,
            outputs=[image_input, question_input, answer_output]
        )

demo.launch(share=True)
