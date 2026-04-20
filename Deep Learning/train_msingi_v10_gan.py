import modal

# Provision the environment with standard ML audio libraries
image = (
    modal.Image.debian_slim()
    .pip_install(
        "jupyterlab",
        "torch",
        "torchaudio",
        "scipy",
        "numpy",
        "IPython"
    )
)

# Attach your specific Sauti project volume
sauti_volume = modal.Volume.from_name("sauti-tts-ckpts")

app = modal.App("sauti-interactive-sandbox")

@app.function(
    image=image, 
    volumes={"/data/sauti-tts-ckpts": sauti_volume}, 
    gpu="T4", # Or "A10G" / "A100" depending on your model size
    timeout=86400 # Keep the session alive for 24 hours
)
@modal.web_server(8888)
def run_jupyter():
    import subprocess
    print("Starting JupyterLab on Modal...")
    # Start JupyterLab without a password/token for easy access via Modal's secure tunnel
    subprocess.Popen([
        "jupyter", "lab", 
        "--ip=0.0.0.0", 
        "--port=8888", 
        "--allow-root", 
        "--no-browser", 
        "--NotebookApp.token=''",
        "--NotebookApp.password=''"
    ])