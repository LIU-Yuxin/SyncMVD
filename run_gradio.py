import gradio as gr

def generate(mesh_file, prompt, neg_prompt, timesteps, guidance_scale):
	print(mesh_file)
	return mesh_file

gr_mesh_input = gr.Model3D(
    camera_position=(0, 180, 1)
)

gr_prompt = gr.Textbox(
	lines=2, 
	placeholder="Prompt", 
	# label="Prompt"
)

gr_neg_prompt = gr.Textbox(
	lines=1, 
	placeholder="Negative Prompt", 
	# label="Negative Prompt"
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

gr_seed = gr.Textbox(
	lines=1, 
	value="-1", 
	label="Seed"
)

gr_controlnet = gr.Dropdown(
	choices=["depth", "normal"], 
	value="depth"
)

gr_camera_azims = gr.Textbox(
	lines=1, 
	value="-180,-135,-90,-45,0,45,90,135", 
	label="Camera azim"
)

gr_mesh_output = gr.Model3D()

gr_generate_btn = gr.Button(value="Generate")






# demo = gr.Interface(
# 	fn = generate,
# 	inputs = [
# 		gr_mesh_input,
# 		gr_prompt,
# 		gr_neg_prompt,
# 		gr_timesteps,
# 		gr_guidance_scale,
# 		gr_seed,
# 		gr_controlnet,
# 		gr_camera_azims
# 	],
# 	outputs=[
# 		gr_mesh_output
# 	]
# )

with gr.Blocks() as demo:
	gr.Markdown(
		"""
		# SyncMVD
		"""
	)
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
		gr_controlnet = gr.Dropdown(
			choices=["depth", "normal"], 
			value="depth"
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
		gr_generate_btn = gr.Button(value="Generate")
		
	with gr.Column():
		gr_mesh_input = gr.Model3D(
		    camera_position=(90, 90	, None)
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

	
	
	gr_mesh_output = gr.Model3D()

	gr_generate_btn.click(
		fn = generate,
		inputs = [
			gr_mesh_input,
			gr_prompt,
			gr_neg_prompt,
			gr_timesteps,
			gr_guidance_scale,
			gr_seed,
			gr_controlnet,
			gr_camera_azims
		],
		outputs=[
			gr_mesh_output
		]
	)
    

if __name__ == '__main__':
	demo.launch()