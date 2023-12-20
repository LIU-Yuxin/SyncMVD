import gradio as gr
import os
from os.path import join, isdir, isfile
import tempfile
from datetime import datetime
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDPMScheduler, UniPCMultistepScheduler
from src.pipeline import StableSyncMVDPipeline
from src.configs import *

def generate(
		mesh_file, 
		prompt, 
		neg_prompt, 
		timesteps, 
		guidance_scale,
		conditioning_scale,
		multiview_diffusion_end,
		seed,
		cond_type,
		camera_azims
	):
	opt = parse_config()
	mesh_path = mesh_file
	output_root = temp_dir

	output_name_components = []
	if opt.prefix and opt.prefix != "":
		output_name_components.append(opt.prefix)
	mesh_name = splitext(basename(mesh_path))[0].replace(" ", "_")
	output_name_components.append(mesh_name)
	output_name_components.append(datetime.now().strftime("%d%b%Y-%H%M%S%f"))
	output_name = "_".join(output_name_components)
	output_dir = join(output_root, output_name)

	if not isdir(output_dir):
		os.mkdir(output_dir)
	else:
		print(f"Results exist in the output directory, use time string to avoid name collision.")
		exit(0)

	print(f"Saving to {output_dir}")

	logging_config = {
		"output_dir":output_dir, 
		# "output_name":None, 
		# "intermediate":False, 
		"log_interval":1000,
		"view_fast_preview": True,
		"tex_fast_preview": True,
		}

	if cond_type == "normal":
		controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_normalbae", variant="fp16", torch_dtype=torch.float16)
	elif cond_type == "depth":
		controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", variant="fp16", torch_dtype=torch.float16)			

	pipe = StableDiffusionControlNetPipeline.from_pretrained(
		"runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
	)


	pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

	syncmvd = StableSyncMVDPipeline(**pipe.components)



	result_tex_rgb, textured_views, v = syncmvd(
		prompt=prompt,
		height=opt.latent_view_size*8,
		width=opt.latent_view_size*8,
		num_inference_steps=timesteps,
		guidance_scale=guidance_scale,
		negative_prompt=negative_prompt,
		
		generator=torch.manual_seed(seed) if seed!=-1 else None,
		max_batch_size=48,
		controlnet_guess_mode=opt.guess_mode,
		controlnet_conditioning_scale = conditioning_scale,
		controlnet_conditioning_end_scale= opt.conditioning_scale_end,
		control_guidance_start= opt.control_guidance_start,
		control_guidance_end = opt.control_guidance_end,
		guidance_rescale = opt.guidance_rescale,
		use_directional_prompt=True,

		mesh_path=mesh_path,
		mesh_transform={"scale":opt.mesh_scale},
		mesh_autouv=opt.mesh_autouv,

		camera_azims=camera_azims,
		top_cameras=opt.top_cameras,
		texture_size=opt.latent_tex_size,
		render_rgb_size=opt.rgb_view_size,
		texture_rgb_size=opt.rgb_tex_size,
		multiview_diffusion_end=multiview_diffusion_end,
		ref_attention_end=opt.ref_attention_end,
		shuffle_background_change=opt.shuffle_bg_change,
		shuffle_background_end=opt.shuffle_bg_end,

		logging_config=logging_config,
		cond_type=cond_type,
		
		
		)
	return [join(output_dir, "results", "textured.obj"), join(output_dir, "results", "textured.png")]


def setup_demo():
	with gr.Blocks() as demo:
		gr.Markdown(
			"""
			# SyncMVD
			generate texture for a 3D object from a text prompt using a Synchronized Multi-View Diffusion approach.
			"""
		)
		with gr.Row():
			
			with gr.Column():
				gr_mesh_input = gr.Model3D(
				    camera_position=(90, 90, None),
				    label="Mesh input"
				)
			
				gr_timesteps = gr.Slider(
					minimum=15,
					maximum=50,
					step=1,
					value=30,
					label="Sampling steps"
				)
				gr_guidance_scale = gr.Slider(
					minimum=1,
					maximum=25,
					step=0.5,
					value=15,
					label="CFG scale"
				)

				gr_conditioning_scale = gr.Slider(
					minimum=0.5,
					maximum=1,
					step=0.05,
					value=0.7,
					label="Conditioning scale"
				)
				gr_multiview_diffusion_end = gr.Slider(
					minimum=0.6,
					maximum=1,
					step=0.05,
					value=0.8,
					label="Multi-view diffusion end"
				)

				gr_generate_btn = gr.Button(value="Generate")

			with gr.Column():
				gr_prompt = gr.Textbox(
					lines=3, 
					placeholder="A photo of a...", 
					label="Prompt"
				)
				gr_neg_prompt  = gr.Textbox(
					lines=3, 
					placeholder="Negative prompt", 
					value="oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect.",
					label="Negative prompt"
				)
				gr_cond_type = gr.Dropdown(
					choices=["depth", "normal"], 
					value="depth",
					label="Conditioning"
				)
				gr_seed = gr.Textbox(
					lines=1, 
					value="-1", 
					label="Seed"
				)
				gr_camera_azims = gr.Textbox(
				lines=1, 
				value="-180,-135,-90,-45,0,45,90,135", 
				label="Camera azim"
				)
				
				
		with gr.Row():
			gr_mesh_output = gr.Model3D(
				camera_position=(90, 90, None),
				label="Textured Mesh"
			)

			gr_texture = gr.Image(
				label="Generated texture"
			)

		gr.Examples(
	        [["data/face/face.obj", "Portrait photo of Kratos, god of war."], ["data/sneaker/sneaker.glb", "A photo of a camouflage military boot."]],
	        inputs=[gr_mesh_input, gr_prompt]
	    )

		gr_generate_btn.click(
			fn = generate,
			inputs = [
				gr_mesh_input,
				gr_prompt,
				gr_neg_prompt,
				gr_timesteps,
				gr_guidance_scale,
				gr_conditioning_scale,
				gr_multiview_diffusion_end,
				gr_seed,
				gr_cond_type,
				gr_camera_azims
			],
			outputs=[
				gr_mesh_output,
				gr_texture
			]
		)

	return demo
    

if __name__ == '__main__':
	with tempfile.TemporaryDirectory() as temp_dir:
		demo = setup_demo()
		demo.launch()