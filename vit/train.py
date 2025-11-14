import argparse
import os
import tensorflow as tf
from transformers import T5Tokenizer

from config import CFG
from data import build_loaders
from modeling_vitt5 import ViTT5ImageCaptioningTF, build_and_freeze
from train_utils import set_seeds, tf_gpu_cleanup, tf_gpu_memory_summary, run_training_optimized

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=CFG.data.dataset, choices=["coco-2017","coco-vn","ktvic"],
                    help="Chọn dataset: coco-2017 | coco-vn | ktvic")
    ap.add_argument("--batch_size", type=int, default=CFG.train.batch_size)
    ap.add_argument("--epochs", type=int, default=CFG.train.num_epochs)
    ap.add_argument("--acc_steps", type=int, default=CFG.train.acc_steps)
    ap.add_argument("--lr_proj", type=float, default=CFG.train.lr_proj)
    ap.add_argument("--lr_t5", type=float, default=CFG.train.lr_t5)
    ap.add_argument("--mixed_precision", action="store_true", default=CFG.train.mixed_precision)
    return ap.parse_args()

def main():
    args = parse_args()
    CFG.data.dataset = args.dataset
    CFG.train.batch_size = args.batch_size
    CFG.train.num_epochs = args.epochs
    CFG.train.acc_steps = args.acc_steps
    CFG.train.lr_proj = args.lr_proj
    CFG.train.lr_t5 = args.lr_t5
    CFG.train.mixed_precision = args.mixed_precision

    set_seeds(42)
    device = "/GPU:0" if len(tf.config.list_physical_devices('GPU')) > 0 else "/CPU:0"
    print(f"[INFO] Device: {device}")

    t5_tokenizer = T5Tokenizer.from_pretrained(CFG.tok.t5_model_name)

    print("[INFO] Building datasets/loaders...")
    train_df, val_df_grouped, train_loader, val_loader = build_loaders(
        CFG.data.dataset, t5_tokenizer, CFG.vis.image_size, CFG.tok.max_length, CFG.train.batch_size
    )
    print("==> Train captions:", len(train_df))
    print("==> Val images:", len(val_df_grouped))

    model = ViTT5ImageCaptioningTF(
        vit_model_name=CFG.model.vit_model_name,
        t5_model_name=CFG.model.t5_model_name,
        embed_dim=None,
        pad_token_id=t5_tokenizer.pad_token_id,
        bos_token_id=getattr(t5_tokenizer, "bos_token_id", None),
        eos_token_id=t5_tokenizer.eos_token_id,
        max_length=CFG.tok.max_length,
        vit_from_pt=CFG.model.vit_from_pt,
        t5_from_pt=CFG.model.t5_from_pt
    )
    model = build_and_freeze(model, pad_token_id=t5_tokenizer.pad_token_id)

    tf_gpu_cleanup(reset_peak=True, enable_memory_growth=True)
    tf_gpu_memory_summary()

    run_training_optimized(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=t5_tokenizer,
        cfg=CFG.train,
        log_cfg=CFG.log,
        vis_cfg=CFG.vis
    )

if __name__ == "__main__":
    main()
