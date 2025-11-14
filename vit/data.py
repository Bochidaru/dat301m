import os, json, re
from typing import List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO
from transformers import T5Tokenizer

from config import CFG

def preprocess_caption(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_coco_data(input_directory: str, split: str) -> pd.DataFrame:
    if split == "train":
        images_dir = os.path.join(input_directory, "train2017")
        annotations_path = os.path.join(input_directory, "annotations/captions_train2017.json")
    elif split == "val":
        images_dir = os.path.join(input_directory, "val2017")
        annotations_path = os.path.join(input_directory, "annotations/captions_val2017.json")
    else:
        raise ValueError("split phải là 'train' hoặc 'val'")

    coco = COCO(annotations_path)
    data = []
    for image_id in coco.getImgIds():
        img_info_list = coco.loadImgs(image_id)
        for img_info in img_info_list:
            img_path = f"{images_dir}/{img_info['file_name']}"
            ann_ids = coco.getAnnIds(imgIds=image_id)
            for ann in coco.loadAnns(ann_ids):
                caption = f"<s> {preprocess_caption(ann['caption'])}</s>"
                data.append({"image": img_path, "caption": caption, "language": "en"})
    return pd.DataFrame(data)

def load_vietnamese_coco_dataset(caption_path: str, base_image_path: str) -> pd.DataFrame:
    dataset_info = [
        ("captions_train2017_trans.json", os.path.join(base_image_path, "train2017")),
        ("captions_val2017_trans.json",   os.path.join(base_image_path, "val2017")),
    ]
    records = []
    for caption_file_name, image_dir in dataset_info:
        caption_file = os.path.join(caption_path, caption_file_name)
        if not os.path.exists(caption_file):
            continue
        with open(caption_file, 'r', encoding="utf-8") as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        image_id_to_file = {
            int(f.split('.')[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir) if f.endswith('.jpg')
        }
        for ann in annotations:
            image_id = ann["image_id"]
            caption  = preprocess_caption(ann["caption"])
            if image_id in image_id_to_file:
                records.append({"image": image_id_to_file[image_id],
                                "caption": caption,
                                "language": "vi"})
    return pd.DataFrame(records)

def load_ktvic_data(input_directory: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(input_directory, "ktvic_dataset/train_data.json")
    test_path  = os.path.join(input_directory, "ktvic_dataset/test_data.json")
    with open(train_path, "r", encoding="utf-8") as f:
        train_json = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_json = json.load(f)

    train_images = {img["id"]: img["filename"] for img in train_json["images"]}
    test_images  = {img["id"]: img["filename"] for img in test_json["images"]}

    train_data, test_data = [], []
    for ann in train_json["annotations"]:
        iid = ann["image_id"]
        if iid in train_images:
            image_path = os.path.join(input_directory, "ktvic_dataset/train-images", train_images[iid])
            train_data.append({"image": image_path,
                               "caption": preprocess_caption(ann["caption"]),
                               "language": "vi"})
    for ann in test_json["annotations"]:
        iid = ann["image_id"]
        if iid in test_images:
            image_path = os.path.join(input_directory, "ktvic_dataset/public-test-images", test_images[iid])
            test_data.append({"image": image_path,
                              "caption": preprocess_caption(ann["caption"]),
                              "language": "vi"})
    df_train = pd.DataFrame(train_data)
    df_test  = pd.DataFrame(test_data)
    df_train = df_train[df_train["image"].apply(os.path.exists)].reset_index(drop=True)
    df_test  = df_test[df_test["image"].apply(os.path.exists)].reset_index(drop=True)
    return df_train, df_test

def load_data(dataset_type: str):
    if dataset_type == "coco-2017":
        return (load_coco_data(CFG.data.coco2017_root, split="train"),
                load_coco_data(CFG.data.coco2017_root, split="val"))
    if dataset_type == "coco-vn":
        return load_vietnamese_coco_dataset(CFG.data.coco_vn_caption_root,
                                            CFG.data.coco2017_root)
    if dataset_type == "ktvic":
        return load_ktvic_data(CFG.data.ktvic_root)
    raise ValueError(f"Unknown dataset_type: {dataset_type}")

def image_transform_fn(image: Image.Image, image_size: int,
                       mean=CFG.vis.mean, std=CFG.vis.std) -> tf.Tensor:
    image = tf.convert_to_tensor(np.array(image), dtype=tf.float32) / 255.0
    image = tf.image.resize(image, (image_size, image_size), method=tf.image.ResizeMethod.BICUBIC)
    mean = tf.constant(mean, dtype=tf.float32)
    std  = tf.constant(std,  dtype=tf.float32)
    image = (image - mean) / std
    return image

def group_captions(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("image")["caption"].apply(list).reset_index()

def make_train_ds(df: pd.DataFrame, tokenizer: T5Tokenizer,
                  image_size: int, max_length: int, batch_size: int) -> tf.data.Dataset:
    def gen():
        for img_path, caption in zip(df["image"], df["caption"]):
            image = Image.open(img_path).convert("RGB")
            image = image_transform_fn(image, image_size)
            tokens = tokenizer(
                caption, truncation=True, padding="max_length",
                max_length=max_length, return_tensors="np")
            yield image, tokens["input_ids"][0].astype(np.int32)
    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(max_length,), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.shuffle(len(df)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def make_val_ds(df_grouped: pd.DataFrame, tokenizer: T5Tokenizer,
                image_size: int, max_length: int, batch_size: int) -> tf.data.Dataset:
    def gen():
        for img_path, captions in zip(df_grouped["image"], df_grouped["caption"]):
            image = Image.open(img_path).convert("RGB")
            image = image_transform_fn(image, image_size)
            token_ids_list = []
            for cap in captions:
                tokens = tokenizer(
                    cap, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors="np")
                token_ids_list.append(tokens["input_ids"][0].astype(np.int32))
            yield image, tf.ragged.constant(token_ids_list, dtype=tf.int32)
    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        tf.RaggedTensorSpec(shape=(None, max_length), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def build_loaders(dataset_type: str, tokenizer: T5Tokenizer,
                  image_size: int, max_length: int, batch_size: int):
    if dataset_type == "coco-vn":
        coco_vn = load_data("coco-vn")
        unique_images = coco_vn["image"].unique()
        train_img_paths, val_img_paths = train_test_split(
            unique_images, test_size=CFG.data.val_ratio_by_image, random_state=42)
        train_df = coco_vn[coco_vn["image"].isin(train_img_paths)].reset_index(drop=True)
        val_df_raw = coco_vn[coco_vn["image"].isin(val_img_paths)].reset_index(drop=True)
        val_df = group_captions(val_df_raw)
        train_loader = make_train_ds(train_df, tokenizer, image_size, max_length, batch_size)
        val_loader   = make_val_ds(val_df, tokenizer, image_size, max_length, batch_size)
        return train_df, val_df, train_loader, val_loader

    if dataset_type == "coco-2017":
        train_df, val_df = load_data("coco-2017") 
        val_df_grouped = group_captions(val_df)
        train_loader = make_train_ds(train_df, tokenizer, image_size, max_length, batch_size)
        val_loader   = make_val_ds(val_df_grouped, tokenizer, image_size, max_length, batch_size)
        return train_df, val_df_grouped, train_loader, val_loader

    if dataset_type == "ktvic":
        train_df, test_df = load_data("ktvic")
        val_df_grouped = group_captions(test_df)
        train_loader = make_train_ds(train_df, tokenizer, image_size, max_length, batch_size)
        val_loader   = make_val_ds(val_df_grouped, tokenizer, image_size, max_length, batch_size)
        return train_df, val_df_grouped, train_loader, val_loader

    raise ValueError(f"Unknown dataset_type: {dataset_type}")
