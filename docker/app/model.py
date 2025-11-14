import os
import copy
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageFile
from timm import create_model as timm_create_model
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ViTT5ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        vit_model_name="vit_base_patch16_224",
        t5_model_name="VietAI/vit5-base",
        embed_dim=None,
        vocab_size=None,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        max_length=30,
    ):
        super(ViTT5ImageCaptioningModel, self).__init__()

        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size

        # --- Vision Encoder (ViT) ---
        self.vit = timm_create_model(vit_model_name, pretrained=True, num_classes=0)
        vit_embed_dim = self.vit.embed_dim

        # --- Text Decoder (T5) ---
        self.t5 = T5Model.from_pretrained(t5_model_name)
        self.t5_for_cond_gen = T5ForConditionalGeneration.from_pretrained(t5_model_name)

        self.t5_embed_dim = self.t5.config.d_model
        # print(f"T5 d_model (embed_dim): {self.t5_embed_dim}")
        # print(f"ViT embed_dim: {vit_embed_dim}")

        if embed_dim is None:
            self.embed_dim = self.t5_embed_dim
        else:
            self.embed_dim = embed_dim

        # Projection from ViT to T5 embedding dimension
        self.vit_projection = nn.Linear(vit_embed_dim, self.embed_dim)

        if self.embed_dim != self.t5_embed_dim:
            raise ValueError(
                f"embed_dim after projection ({self.embed_dim}) must match T5's d_model ({self.t5_embed_dim})."
            )

    def forward(self, images, captions=None):
        batch_size = images.size(0)

        # --- Image Encoding ---
        vit_features = self.vit.forward_features(
            images
        )  # (B, num_patches, vit_embed_dim)
        projected_vit_features = self.vit_projection(
            vit_features
        )  # (B, num_patches, embed_dim)

        # --- Encoder Outputs ---
        start_token_id = (
            self.bos_token_id if self.bos_token_id is not None else self.pad_token_id
        )
        if start_token_id is None:
            raise ValueError(
                "Need bos_token_id or pad_token_id to create dummy input_ids."
            )

        dummy_input_ids = torch.full(
            (batch_size, 1),
            start_token_id,
            dtype=torch.long,
            device=images.device,
        )

        encoder_outputs = self.t5.encoder(input_ids=dummy_input_ids, return_dict=True)
        encoder_outputs.last_hidden_state = projected_vit_features

        if captions is not None:
            # Training mode
            outputs = self.t5_for_cond_gen(
                input_ids=captions,
                encoder_outputs=encoder_outputs,
                labels=captions,
            )
            return outputs.loss
        else:
            # Inference mode
            try:
                generated_ids = self.t5_for_cond_gen.generate(
                    inputs=dummy_input_ids,
                    encoder_outputs=encoder_outputs,
                    max_length=self.max_length,
                    num_beams=3,
                    early_stopping=True,
                    pad_token_id=self.pad_token_id,
                    bos_token_id=self.bos_token_id,
                    eos_token_id=self.eos_token_id,
                )
            except Exception as e:
                print(f"Error during generation: {e}")
                raise e
            return generated_ids


def get_tokenizer(model_name: str = "VietAI/vit5-base"):
    return T5Tokenizer.from_pretrained(model_name, legacy=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_model(
    tokenizer: T5Tokenizer,
    device: torch.device,
    vit_model_name: str = "vit_base_patch16_224",
    t5_model_name: str = "VietAI/vit5-base",
    max_length: int = 35,
):
    model = ViTT5ImageCaptioningModel(
        vit_model_name=vit_model_name,
        t5_model_name=t5_model_name,
        embed_dim=None,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
    )
    model.to(device)
    return model


def load_model_weights_from_path(model: nn.Module, path: str, device: torch.device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    return model


def generate_caption(model, image_path, tokenizer, transform, device, max_length=35):
    """
    Generate a caption for a single image.
    """
    model.eval()
    with torch.no_grad():
        # 1. Load and preprocess image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        # 2. Generate caption
        generated_ids = model(image_tensor)

        # 3. Decode token IDs to text
        if generated_ids is not None:
            generated_caption = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
            return generated_caption
        else:
            return None
        

if __name__ == "__main__":
    image_path = (
        "./img.jpg"
    )
    # print(f"Generating caption for image: {image_path}")
    # Example standalone run (will instantiate fresh components)
    tokenizer = get_tokenizer()
    device = get_device()
    transform = get_image_transform()
    model = build_model(tokenizer=tokenizer, device=device, max_length=35)
    # Provide a local path to weights if available
    load_model_weights_from_path(model, "./weights/best_model_weights.pth", device)
    caption = generate_caption(
        model, image_path, tokenizer, transform, device, max_length=30
    )
    if caption:
        print(f"Generated caption: {caption}")
    else:
        print("Failed to generate caption.")
