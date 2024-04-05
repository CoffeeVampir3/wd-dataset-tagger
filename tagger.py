import numpy as np
import pandas as pd
import timm
import os, glob, sys
import torch
from huggingface_hub import hf_hub_download
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn
from torch.nn import functional as F
from PIL import Image
import json
from dataclasses import dataclass

def load_cfg_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)

def load_cfg(cfg_path):
    hf_config = load_cfg_from_json(cfg_path)
    if 'pretrained_cfg' not in hf_config:
        # old form, pull pretrain_cfg out of the base dict
        pretrained_cfg = hf_config
        hf_config = {}
        hf_config['architecture'] = pretrained_cfg.pop('architecture')
        hf_config['num_features'] = pretrained_cfg.pop('num_features', None)
        if 'labels' in pretrained_cfg:  # deprecated name for 'label_names'
            pretrained_cfg['label_names'] = pretrained_cfg.pop('labels')
        hf_config['pretrained_cfg'] = pretrained_cfg
    
    pretrained_cfg = hf_config['pretrained_cfg']
    pretrained_cfg['hf_hub_id'] = hf_config["architecture"]  # insert hf_hub id for pretrained weight load during model creation
    pretrained_cfg['source'] = 'hf-hub'
    
    # model should be created with base config num_classes if its exist
    if 'num_classes' in hf_config:
        pretrained_cfg['num_classes'] = hf_config['num_classes']
    
    # label meta-data in base config overrides saved pretrained_cfg on load
    if 'label_names' in hf_config:
        pretrained_cfg['label_names'] = hf_config.pop('label_names')
    if 'label_descriptions' in hf_config:
        pretrained_cfg['label_descriptions'] = hf_config.pop('label_descriptions')
    
    model_args = hf_config.get('model_args', {})
    model_name = hf_config['architecture']

    return pretrained_cfg, model_name, model_args

#"/home/blackroot/Desktop/ViTagger/model/config.json"
def load_model_but_good_this_time_srsly(model_path, model_config):
    pretrained_cfg, model_name, model_args = load_cfg(model_config)
    model = timm.create_model(
        model_name, 
        checkpoint_path=model_path, 
        pretrained_cfg=pretrained_cfg,
        **model_args)

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform

@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]
def get_tag_frame(dataset_path):
    df: pd.DataFrame = pd.read_csv(dataset_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data

def convert_probs_to_tags(
    probs: Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs.numpy()))

    # First 4 labels are actually ratings
    rating_labels = dict([probs[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\(").replace(")", "\)")

    return caption, taglist, rating_labels, char_labels, gen_labels

def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image

def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas

def tag_image(model, transform, image):
    img_input = pil_ensure_rgb(image)
    img_input = pil_pad_square(img_input)
    inputs: Tensor = transform(img_input).unsqueeze(0)
    inputs = inputs[:, [2, 1, 0]]

    with torch.inference_mode():
        model = model.to("cuda")
        inputs = inputs.to("cuda")
        # run the model
        outputs = model.forward(inputs)
        # apply the final activation function (timm doesn't support doing this internally)
        outputs = F.sigmoid(outputs)
        # move inputs, outputs, and model back to to cpu if we were on GPU
        del inputs
        outputs = outputs.to("cpu")

        return outputs

def judge_probs(labels, probs, filename):
    caption, taglist, ratings, character, general = convert_probs_to_tags(
        probs=probs.squeeze(0),
        labels=labels,
        gen_threshold=0.35,
        char_threshold=0.75,
    )
    # print("--------")
    # print(f"Caption: {caption}")
    # print("--------")
    # print(f"Tags: {taglist}")

    # print("--------")
    # print("Ratings:")
    # for k, v in ratings.items():
    #     print(f"  {k}: {v:.3f}")

    # print("--------")
    # print(f"Character tags (threshold={0.35}):")
    # for k, v in character.items():
    #     print(f"  {k}: {v:.3f}")

    # print("--------")
    # print(f"General tags (threshold={0.75}):")
    # for k, v in general.items():
    #     print(f"  {k}: {v:.3f}")

    comma_seperated_tags = ', '.join([item[0] for item in general.items()])

    with open(filename, "w") as file:
        file.write(comma_seperated_tags)

def as_txt_extension(filename):
    directory, filename_with_ext = os.path.split(filename)
    filename, _ = os.path.splitext(filename_with_ext)
    
    # Construct the new file path with the .txt extension
    new_file_path = os.path.join(directory, filename + ".txt")
    
    return new_file_path

def get_file_paths(directory):
    file_paths = {}
    
    for extension in ["*.json", "*.csv", "*.safetensors"]:
        files = glob.glob(os.path.join(directory, extension))
        
        if files:
            file_path = files[0]
            extension_without_dot = extension[1:]
            file_paths[extension_without_dot] = file_path
    
    return file_paths[".safetensors"], file_paths[".json"], file_paths[".csv"]

def iterate_images(directory):
    image_ext = ['.png', '.jpg', '.jpeg']
    
    for extension in image_ext:
        for file_path in glob.glob(os.path.join(directory, '*' + extension)):
            yield file_path

model_path, config_path, tag_frame_path = get_file_paths("model")
    
model, transform = load_model_but_good_this_time_srsly(model_path, config_path)
tag_df = get_tag_frame(tag_frame_path)

dir_path = sys.argv[1]
for image_path in iterate_images(dir_path):
    tag_pair_path = as_txt_extension(image_path)
    image = Image.open(image_path)
    tag_probs = tag_image(model, transform, image)
    judge_probs(tag_df, tag_probs, tag_pair_path)