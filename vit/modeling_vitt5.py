import re
from typing import List, Tuple, Dict
import tensorflow as tf
from transformers import (
    TFViTModel,
    TFT5Model,
    TFT5ForConditionalGeneration,
    AutoTokenizer,
    AutoImageProcessor,
)
from transformers.modeling_tf_outputs import TFBaseModelOutput

def flatten_layers(layer):
    return list(layer._flatten_layers(recursive=True, include_self=False))

def pick_layers_by_name(root_layer, substrings):
    subs = [s.lower() for s in substrings]
    picked = []
    for L in flatten_layers(root_layer):
        name = (L.name or "").lower()
        if any(s in name for s in subs):
            picked.append(L)
    return picked

def var_key(v):
    kfn = getattr(v, "experimental_ref", None)
    return kfn() if callable(kfn) else id(v)

def dedupe(vars_list):
    seen, uniq = set(), []
    for v in vars_list:
        k = var_key(v)
        if k not in seen:
            seen.add(k); uniq.append(v)
    return uniq

def vars_of_layers(layers):
    vs = []
    for L in layers:
        vs.extend(getattr(L, "variables", []))
    return vs

def trainable_vars_of_layers(layers):
    vs = []
    for L in layers:
        vs.extend(getattr(L, "trainable_variables", []))
    return vs

def num_params(vs):
    return sum(int(v.shape.num_elements()) for v in vs)

def is_shared_embedding_layer(L, d_model, min_vocab=10000):
    nm = (L.name or "").lower()
    if ("shared" not in nm) and ("embed" not in nm):
        return False
    for v in getattr(L, "variables", []):
        shp = v.shape.as_list()
        if shp and len(shp) == 2 and shp[1] == d_model and (shp[0] or 0) >= min_vocab:
            return True
    return False

def find_shared_layers(root_layer, d_model):
    return [L for L in flatten_layers(root_layer) if is_shared_embedding_layer(L, d_model)]

class ViTT5ImageCaptioningTF(tf.keras.Model):
    def __init__(
        self,
        vit_model_name="google/vit-base-patch16-224",
        t5_model_name="VietAI/vit5-base",
        embed_dim=None,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        max_length=35,
        vit_from_pt=False,
        t5_from_pt=True
    ):
        super().__init__()
        self.max_length   = max_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        try:
            self.vit = TFViTModel.from_pretrained(vit_model_name, from_pt=vit_from_pt)
        except Exception:
            self.vit = TFViTModel.from_pretrained(vit_model_name, from_pt=True)
        vit_embed_dim = self.vit.config.hidden_size

        try:
            self.t5 = TFT5Model.from_pretrained(t5_model_name, from_pt=t5_from_pt)
            self.t5_for_cond_gen = TFT5ForConditionalGeneration.from_pretrained(
                t5_model_name, from_pt=t5_from_pt
            )
        except Exception:
            self.t5 = TFT5Model.from_pretrained(t5_model_name, from_pt=True)
            self.t5_for_cond_gen = TFT5ForConditionalGeneration.from_pretrained(
                t5_model_name, from_pt=True
            )

        if hasattr(self, "_track_trackable"):
            self._track_trackable(self.t5, name="t5_plain")
            self._track_trackable(self.t5_for_cond_gen, name="t5_condgen")

        self.t5_embed_dim = int(self.t5.config.d_model)
        self.embed_dim = self.t5_embed_dim if embed_dim is None else int(embed_dim)
        if self.embed_dim != self.t5_embed_dim:
            raise ValueError(
                f"embed_dim sau projection ({self.embed_dim}) phải bằng T5 d_model ({self.t5_embed_dim})."
            )

        self.vit_projection = tf.keras.layers.Dense(self.embed_dim, name="vit_to_t5_projection")

        self._plain_dec_layers = None
        self._cg_dec_layers    = None
        self._plain_shared     = None
        self._cg_shared        = None

    def call(self, images, captions=None, training=False, return_loss=False):
        def _maybe_transpose(imgs):
            cond = tf.logical_and(tf.equal(tf.rank(imgs), 4),
                                  tf.logical_and(tf.not_equal(tf.shape(imgs)[1], 3),
                                                 tf.equal(tf.shape(imgs)[-1], 3)))
            return tf.cond(cond, lambda: tf.transpose(imgs, [0,3,1,2]), lambda: imgs)
        images_bchw = _maybe_transpose(images)

        B = tf.shape(images_bchw)[0]
        vit_out = self.vit(pixel_values=images_bchw, training=training, return_dict=True)
        proj    = self.vit_projection(vit_out.last_hidden_state)

        enc_out = TFBaseModelOutput(last_hidden_state=proj)
        start_id = self.bos_token_id if self.bos_token_id is not None else self.pad_token_id
        if start_id is None:
            raise ValueError("Cần bos_token_id hoặc pad_token_id để tạo dummy inputs.")
        dummy_inputs = tf.fill([B, 1], tf.cast(start_id, tf.int32))

        if captions is not None:
            out = self.t5_for_cond_gen(
                input_ids=captions,
                encoder_outputs=enc_out,
                labels=captions,
                training=training,
            )
            return out.loss if return_loss else out

        gen = self.t5_for_cond_gen.generate(
            inputs=dummy_inputs,
            encoder_outputs=enc_out,
            max_length=int(self.max_length),
            num_beams=3,
            early_stopping=True,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )
        return gen

    def set_train_groups(self, t5_plain_dec_layers, t5cg_dec_layers,
                         t5_plain_shared_layers, t5cg_shared_layers):
        self._plain_dec_layers = t5_plain_dec_layers
        self._cg_dec_layers    = t5cg_dec_layers
        self._plain_shared     = t5_plain_shared_layers
        self._cg_shared        = t5cg_shared_layers

    def selected_trainable_variables(self):
        assert self._plain_dec_layers is not None, "Chưa gọi set_train_groups(...)"
        proj_vars = dedupe(self.vit_projection.trainable_variables)

        dec_plain_vars = dedupe(trainable_vars_of_layers(self._plain_dec_layers))
        dec_cg_vars    = dedupe(trainable_vars_of_layers(self._cg_dec_layers))

        shared_plain   = set(var_key(v) for v in dedupe(vars_of_layers(self._plain_shared)))
        shared_cg      = set(var_key(v) for v in dedupe(vars_of_layers(self._cg_shared)))

        def minus_shared(vs, shared_keys):
            return dedupe([v for v in vs if var_key(v) not in shared_keys])

        dec_plain_vars = minus_shared(dec_plain_vars, shared_plain)
        dec_cg_vars    = minus_shared(dec_cg_vars,    shared_cg)

        return dedupe(proj_vars + dec_plain_vars + dec_cg_vars)

def build_and_freeze(model: ViTT5ImageCaptioningTF, pad_token_id: int):
    B, H, W = 2, 224, 224
    dummy_images_bchw = tf.zeros((B, 3, H, W), dtype=tf.float32)
    vit_out = model.vit(pixel_values=dummy_images_bchw, training=False, return_dict=True)
    proj    = model.vit_projection(vit_out.last_hidden_state)

    _ = model.t5(
        input_ids=tf.fill([B, 5], tf.constant(pad_token_id, dtype=tf.int32)),
        decoder_input_ids=tf.fill([B, 6], tf.constant(pad_token_id, dtype=tf.int32)),
        training=False, return_dict=True,
    )
    dummy_caps = tf.fill([B, 8], tf.constant(pad_token_id, dtype=tf.int32))
    _ = model.t5_for_cond_gen(
        input_ids=dummy_caps,
        encoder_outputs=TFBaseModelOutput(last_hidden_state=proj),
        labels=dummy_caps, training=True,
    )

    model.vit.trainable = False
    model.vit_projection.trainable = True

    t5_plain_enc_layers    = pick_layers_by_name(model.t5,              ["encoder"])
    t5_plain_dec_layers    = pick_layers_by_name(model.t5,              ["decoder"])
    t5_plain_shared_layers = find_shared_layers (model.t5,              model.t5_embed_dim)

    t5cg_enc_layers        = pick_layers_by_name(model.t5_for_cond_gen, ["encoder"])
    t5cg_dec_layers        = pick_layers_by_name(model.t5_for_cond_gen, ["decoder"])
    t5cg_shared_layers     = find_shared_layers (model.t5_for_cond_gen, model.t5_embed_dim)
    t5cg_lm_head_layers    = pick_layers_by_name(model.t5_for_cond_gen, ["lm_head"])

    for L in t5_plain_enc_layers + t5cg_enc_layers + t5cg_lm_head_layers:
        L.trainable = False
    for L in t5_plain_dec_layers + t5cg_dec_layers:
        L.trainable = True
    for L in t5_plain_shared_layers + t5cg_shared_layers:
        L.trainable = False
        for v in getattr(L, "variables", []):
            if hasattr(v, "_trainable"):
                v._trainable = False

    model.set_train_groups(
        t5_plain_dec_layers, t5cg_dec_layers,
        t5_plain_shared_layers, t5cg_shared_layers
    )
    return model
