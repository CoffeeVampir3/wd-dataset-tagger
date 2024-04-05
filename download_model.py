from huggingface_hub import snapshot_download, hf_hub_download
import os

def download_model_data(directory):
    repo = "SmilingWolf/wd-vit-tagger-v3"
    os.makedirs(directory, exist_ok=True)

    hf_hub_download(
        repo_id=repo, 
        filename="model.safetensors",
        local_dir=directory,
        local_dir_use_symlinks=False)

    hf_hub_download(
        repo_id=repo, 
        filename="selected_tags.csv",
        local_dir=directory,
        local_dir_use_symlinks=False)

    hf_hub_download(
        repo_id=repo, 
        filename="config.json",
        local_dir=directory,
        local_dir_use_symlinks=False)