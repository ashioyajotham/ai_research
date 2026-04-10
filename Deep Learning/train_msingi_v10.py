
import modal
import os

app = modal.App("sauti-ya-kenya-v10")
vol = modal.Volume.from_name("sauti-tts-volume")

v10_final_image = (
    modal.Image.debian_slim()
    .apt_install("libsndfile1", "ffmpeg") 
    .pip_install(
        "torch", "transformers", "librosa", "soundfile", 
        "bigvgan", "wandb", "tqdm", "torchaudio", 
        "huggingface_hub", "auraloss", "pesq", "pystoi"
    )
)

class AttrDict(dict):
    def __getattr__(self, name): return self[name]

def get_msingi_mel(wav, device):
    import torch, torchaudio
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, n_fft=1024, win_length=1024,
        hop_length=256, f_min=0, f_max=12000, n_mels=100,
        center=False, power=1.0, normalized=False
    ).to(device)
    mel = transform(wav)
    return torch.log(torch.clamp(mel, min=1e-5))

@app.function(
    image=v10_final_image,
    volumes={"/mnt/sauti": vol},
    gpu="RTX-PRO-6000", 
    memory=128000, 
    timeout=64800, # 18 Hours
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")] 
)
def train_v10_final_sprint():
    import tarfile, io, json, os, torch, librosa, wandb, auraloss
    from bigvgan import BigVGAN
    from tqdm import tqdm
    from huggingface_hub import hf_hub_download

    # 1. Start WandB with a "Stability" tag
    wandb.init(project="sauti-ya-kenya-v10", name="v15-lr-5e-5-stabilization")
    device = "cuda"

    # 2. Assemble Model
    print("🏗️  Assembling BigVGAN-v2...")
    repo_id = "nvidia/bigvgan_v2_24khz_100band_256x"
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="bigvgan_generator.pt")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path, "r") as f:
        h = AttrDict(json.load(f))
    model = BigVGAN(h).to(device)

    # 3. Resume Logic
    checkpoint_path = "/mnt/sauti/v10/msingi_tokens/checkpoints/v10_vocoder_64000.pt"
    if os.path.exists(checkpoint_path):
        print(f"♻️  Found 64k checkpoint! Resuming Msingi v10.5...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        success_count = 64000
    else:
        print("🆕  No checkpoint found, starting from base weights.")
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['generator'])
        success_count = 0

    model.train()
    stft_loss_fn = auraloss.freq.MultiResolutionSTFTLoss().to(device)

    # --- REDUCED LEARNING RATE ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5) 
    # -----------------------------

    # 4. RAM Load
    tar_path = "/mnt/sauti/v10/data/raw/audio/phase1_foundation/common-voice-scripted-speech-25-0-swahil-dfb28a71.tar.gz"
    print("🚀 Loading foundation into RAM...")
    with open(tar_path, "rb") as f:
        ram_buffer = io.BytesIO(f.read())

    # 5. Index
    manifest_path = "/mnt/sauti/v10/data/processed/phase1_manifest_sorted.jsonl"
    validated_samples = {}
    with open(manifest_path, "r") as f:
        for line in f:
            s = json.loads(line)
            validated_samples[os.path.basename(s["audio_path"])] = s["msingi_tokens"]

    # 6. Training Loop
    print("🌊 Starting Stabilization Sprint...")
    with tarfile.open(fileobj=ram_buffer, mode="r:gz") as tar:
        pbar = tqdm(total=len(validated_samples), initial=success_count, desc="v10 Training")
        for member in tar:
            if not member.isfile(): continue
            member_key = os.path.basename(member.name)

            if member_key in validated_samples:
                try:
                    audio_bytes = tar.extractfile(member).read()
                    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=24000)

                    wav = torch.FloatTensor(audio).unsqueeze(0).to(device)
                    mel = get_msingi_mel(wav, device)
                    y_hat = model(mel)

                    t_len = min(y_hat.shape[-1], wav.shape[-1])
                    loss = stft_loss_fn(y_hat[:, :, :t_len], wav[:, :t_len])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    success_count += 1
                    pbar.update(1)

                    if success_count % 10 == 0:
                        wandb.log({"STFT_Loss": loss.item(), "step": success_count})

                    if success_count % 500 == 0:
                        save_path = f"/mnt/sauti/v10/msingi_tokens/checkpoints/v10_vocoder_{success_count}.pt"
                        torch.save(model.state_dict(), save_path)
                        vol.commit()

                except Exception:
                    continue

    vol.commit()
    wandb.finish()

if __name__ == "__main__":
    with app.run():
            train_v10_final_sprint.remote()
