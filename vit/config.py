from dataclasses import dataclass

@dataclass
class DataConfig:
    dataset: str = "coco-vn"
    coco2017_root: str = "/kaggle/input/coco-2017-dataset/coco2017"
    coco_vn_caption_root: str = "/kaggle/input/vietnamese-coco-2017-image-caption-dataset"
    ktvic_root: str = "/kaggle/input/ktvic-dataset"
    val_ratio_by_image: float = 0.025 

@dataclass
class TokenizerConfig:
    t5_model_name: str = "VietAI/vit5-base"
    max_length: int = 35

@dataclass
class VisionConfig:
    image_size: int = 224  
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

@dataclass
class TrainConfig:
    batch_size: int = 32        
    num_epochs: int = 10
    acc_steps: int = 8           
    lr_proj: float = 3e-4
    lr_t5: float = 5e-5
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    early_patience: int = 5
    warmup_frac: float = 0.08  
    mixed_precision: bool = True
    beams_eval: int = 1          

@dataclass
class LogConfig:
    run_name_prefix: str = "run"
    save_max_samples: int = 6

@dataclass
class ModelConfig:
    vit_model_name: str = "google/vit-base-patch16-224"
    t5_model_name: str = "VietAI/vit5-base"
    vit_from_pt: bool = False
    t5_from_pt: bool = True

@dataclass
class CFG:
    data = DataConfig()
    tok  = TokenizerConfig()
    vis  = VisionConfig()
    train= TrainConfig()
    log  = LogConfig()
    model= ModelConfig()
