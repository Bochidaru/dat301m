import os, io, csv, math, datetime, shutil, subprocess, gc
from typing import List, Tuple
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from modeling_vitt5 import var_key, dedupe, vars_of_layers
from eval_utils import evaluate_model_tf, preprocess_caption_for_eval

def set_seeds(seed=42):
    import numpy as np, random, os
    tf.random.set_seed(seed); np.random.seed(seed); random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def _bytes_to_gib(b): return float(b) / (1024 ** 3)

def tf_gpu_memory_summary():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("CUDA is not available.")
        return
    print("GPU Memory Summary (TensorFlow allocator):")
    for idx, _ in enumerate(gpus):
        dev = f"GPU:{idx}"
        try:
            info = tf.config.experimental.get_memory_info(dev)
            cur_gib  = _bytes_to_gib(info.get('current', 0))
            peak_gib = _bytes_to_gib(info.get('peak', 0))
            print(f"  {dev} -> Current: {cur_gib:.2f} GiB | Peak: {peak_gib:.2f} GiB")
        except Exception as e:
            print(f"  {dev} -> (memory_info not available): {e}")

    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                encoding="utf-8"
            ).strip().splitlines()
            print("\nGPU Memory Summary (nvidia-smi):")
            for line in out:
                idx, name, used_mb, total_mb = [x.strip() for x in line.split(",")]
                used_gib  = float(used_mb) / 1024.0
                total_gib = float(total_mb) / 1024.0
                print(f"  GPU:{idx} {name} -> Used: {used_gib:.2f}/{total_gib:.2f} GiB")
        except Exception as e:
            print(f"(nvidia-smi check failed): {e}")

def tf_gpu_cleanup(reset_peak=True, enable_memory_growth=False):
    gpus = tf.config.list_physical_devices('GPU')
    if enable_memory_growth and gpus:
        for g in gpus:
            try: tf.config.experimental.set_memory_growth(g, True)
            except Exception: pass
    try: tf.keras.backend.clear_session()
    except Exception: pass
    gc.collect()
    if reset_peak and gpus:
        for idx, _ in enumerate(gpus):
            dev = f"GPU:{idx}"
            try: tf.config.experimental.reset_memory_stats(dev)
            except Exception: pass

class CosineWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = float(total_steps)
        self.warmup_steps = float(warmup_steps)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        ws = tf.constant(self.warmup_steps, tf.float32)
        ts = tf.constant(self.total_steps, tf.float32)
        warm = tf.minimum(1.0, step / tf.maximum(1.0, ws))
        prog = tf.clip_by_value((step - ws) / tf.maximum(1.0, ts - ws), 0.0, 1.0)
        cosv = 0.5 * (1.0 + tf.cos(np.pi * prog))
        mult = tf.where(step < ws, warm, cosv)
        return self.base_lr * mult

class EarlyStoppingTF:
    def __init__(self, patience=2, path='best_model_weights.h5', verbose=True, delta=0.0):
        self.patience=patience; self.path=path; self.verbose=verbose; self.delta=float(delta)
        self.best=None; self.wait=0; self.early=False
    def __call__(self, val_loss, model):
        s = -float(val_loss)
        if self.best is None or s>self.best+self.delta:
            self.best=s; self.wait=0
            if self.verbose: print(f"[EarlyStopping] save {self.path}")
            model.save_weights(self.path)
        else:
            self.wait+=1
            if self.verbose: print(f"[EarlyStopping] no improve {self.wait}/{self.patience}")
            if self.wait>=self.patience: self.early=True

def inverse_normalize(imgs, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    mean = np.array(mean, np.float32)
    std  = np.array(std,  np.float32)
    return np.clip(imgs * std + mean, 0, 1)

def save_samples(model, val_loader, tokenizer, out_dir, epoch_idx, step_idx, k=6):
    for images, ragged_refs in val_loader.take(1):
        images = tf.cast(images, tf.float32)
        gen_ids = model(images=images, training=False)
        preds = tokenizer.batch_decode(gen_ids.numpy(), skip_special_tokens=True)
        refs = [[tokenizer.decode(x, skip_special_tokens=True) for x in r]
                for r in ragged_refs.to_list()]
        m = min(k, images.shape[0])
        imgs = inverse_normalize(images.numpy()[:m])
        cols = min(3, m); rows = int(math.ceil(m/cols))
        plt.figure(figsize=(4*cols, 4.5*rows))
        for i in range(m):
            ax = plt.subplot(rows, cols, i+1); ax.imshow(imgs[i]); ax.axis("off")
            ax.set_title(f"Pred: {preds[i]}\nRef: {refs[i][0] if refs[i] else ''}",
                         fontsize=9, loc="left", wrap=True)
        os.makedirs(out_dir, exist_ok=True)
        outp = os.path.join(out_dir, f"samples_ep{epoch_idx:03d}_st{int(step_idx)}.png")
        plt.tight_layout(); plt.savefig(outp, dpi=150); plt.close()
        print(f"[Saved samples] {outp}")
        break

def plot_curves(history, out_path):
    eps=[x["epoch"] for x in history]
    tr =[x["train_loss"] for x in history]
    vl =[x["val_loss"] for x in history]
    bl =[x["bleu4"] for x in history]
    plt.figure(figsize=(9,4.6))
    plt.subplot(1,2,1); plt.plot(eps,tr,label="Train"); plt.plot(eps,vl,label="Val"); plt.grid(True,alpha=.3); plt.title("Loss"); plt.legend()
    plt.subplot(1,2,2); plt.plot(eps,bl,label="BLEU-4"); plt.grid(True,alpha=.3); plt.title("BLEU-4"); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()

def run_training_optimized(model,
                           train_loader,
                           val_loader,
                           tokenizer,
                           cfg,
                           log_cfg,
                           vis_cfg):
    if cfg.mixed_precision:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision: mixed_float16")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            try: tf.config.experimental.set_memory_growth(g, True)
            except Exception: pass

    steps_per_epoch = math.ceil(len(list(train_loader.unbatch().as_numpy_iterator())) / cfg.batch_size) \
                      if not hasattr(train_loader, "__len__") else len(train_loader)
    total_steps  = cfg.num_epochs * steps_per_epoch
    warmup_steps = max(100, int(cfg.warmup_frac * total_steps))
    print(f"[INFO] steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, warmup={warmup_steps}")

    sched_proj = CosineWithWarmup(cfg.lr_proj, total_steps, warmup_steps)
    sched_t5   = CosineWithWarmup(cfg.lr_t5,   total_steps, warmup_steps)
    global_step = tf.Variable(0, dtype=tf.int64)

    if not model.vit_projection.variables:
        _ = model(images=tf.zeros((1, vis_cfg.image_size, vis_cfg.image_size, 3), tf.float32),
                  captions=tf.fill([1, tokenizer.model_max_length], tf.cast(tokenizer.pad_token_id, tf.int32)),
                  training=False, return_loss=True)

    def vkey(v):
        kfn = getattr(v, "experimental_ref", None)
        return kfn() if callable(kfn) else id(v)

    def group_decay(vs):
        decay, nodecay = [], []
        for v in vs:
            n = v.name.lower()
            is_bias_or_ln = (n.endswith("/bias:0") or n.endswith("/beta:0") or n.endswith("/gamma:0")
                             or "layer_norm" in n or "layernorm" in n or "layer-normalization" in n)
            (nodecay if is_bias_or_ln else decay).append(v)
        return decay, nodecay

    proj_vars   = dedupe(model.vit_projection.trainable_variables)
    cg_dec_vars = dedupe([v for L in model._cg_dec_layers for v in L.trainable_variables])

    proj_decay, proj_nodecay = group_decay(proj_vars)
    t5_decay,   t5_nodecay   = group_decay(cg_dec_vars)
    sel_vars_for_grads = dedupe(proj_vars + cg_dec_vars)

    def nparams(vs): return sum(int(v.shape.num_elements()) for v in vs)

    opt_proj = tf.keras.optimizers.Adam(learning_rate=sched_proj, beta_1=0.9, beta_2=0.999, epsilon=1e-8, name="adam_proj")
    opt_t5   = tf.keras.optimizers.Adam(learning_rate=sched_t5,   beta_1=0.9, beta_2=0.999, epsilon=1e-8, name="adam_t5")

    def _suffix(n: str): return n.split("/", 1)[1] if "/" in n else n
    _plain_dec_all = dedupe([v for L in model._plain_dec_layers for v in L.variables])
    _cg_dec_all    = dedupe([v for L in model._cg_dec_layers    for v in L.variables])
    _plain_map = { _suffix(v.name): v for v in _plain_dec_all }
    _cg_map    = { _suffix(v.name): v for v in _cg_dec_all    }

    @tf.function
    def sync_plain_from_condgen():
        cnt = 0
        for key, vsrc in _cg_map.items():
            vdst = _plain_map.get(key, None)
            if vdst is None: continue
            if vdst.shape == vsrc.shape:
                vdst.assign(tf.cast(vsrc, vdst.dtype))
                cnt += 1
        return cnt

    accum_vars = [tf.Variable(tf.zeros_like(v), trainable=False) for v in sel_vars_for_grads]
    accum_step = tf.Variable(0, dtype=tf.int32)

    @tf.function(jit_compile=False)
    def train_step_acc(images, captions_ids):
        with tf.GradientTape() as tape:
            loss = model(images=images, captions=captions_ids, training=True, return_loss=True)
        grads = tape.gradient(loss, sel_vars_for_grads)
        grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, sel_vars_for_grads)]
        for a, g in zip(accum_vars, grads):
            a.assign_add(g)
        accum_step.assign_add(1)
        return loss

    @tf.function(jit_compile=False)
    def apply_accumulated_grads():
        def pick(list_vars):
            ids = set(int(vkey(v)) for v in list_vars)
            gs, vs = [], []
            for g, v in zip(accum_vars, sel_vars_for_grads):
                if int(vkey(v)) in ids:
                    gs.append(g); vs.append(v)
            return gs, vs

        gpd, vpd = pick(proj_decay)
        gpn, vpn = pick(proj_nodecay)
        gtd, vtd = pick(t5_decay)
        gtn, vtn = pick(t5_nodecay)

        if cfg.acc_steps > 1:
            scale = lambda L: [x / tf.cast(cfg.acc_steps, x.dtype) for x in L]
            gpd, gpn, gtd, gtn = scale(gpd), scale(gpn), scale(gtd), scale(gtn)

        if gpd or gpn:
            allg = gpd + gpn
            allg, _ = tf.clip_by_global_norm(allg, cfg.grad_clip_norm)
            gpd, gpn = allg[:len(gpd)], allg[len(gpd):]
        if gtd or gtn:
            allg = gtd + gtn
            allg, _ = tf.clip_by_global_norm(allg, cfg.grad_clip_norm)
            gtd, gtn = allg[:len(gtd)], allg[len(gtd):]

        if gpd or gpn:
            opt_proj.apply_gradients(list(zip(gpd, vpd)) + list(zip(gpn, vpn)))
            lr_p = opt_proj.learning_rate(global_step)
            for v in vpd: 
                v.assign_sub(lr_p * cfg.weight_decay * tf.cast(v, v.dtype))
        if gtd or gtn:
            opt_t5.apply_gradients(list(zip(gtd, vtd)) + list(zip(gtn, vtn)))
            lr_t = opt_t5.learning_rate(global_step)
            for v in vtd:
                v.assign_sub(lr_t * cfg.weight_decay * tf.cast(v, v.dtype))

        for a in accum_vars: a.assign(tf.zeros_like(a))
        accum_step.assign(0)
        _ = sync_plain_from_condgen()

    RUN_NAME = datetime.datetime.now().strftime(f"{log_cfg.run_name_prefix}_%Y%m%d-%H%M%S")
    LOG_DIR   = os.path.join("./runs/vitt5_tf", RUN_NAME)
    CKPT_DIR  = os.path.join(LOG_DIR, "checkpoints")
    WEIGHT_DIR= os.path.join(LOG_DIR, "weights_by_epoch")
    FIG_DIR   = os.path.join(LOG_DIR, "figs")
    SAMPLE_DIR= os.path.join(LOG_DIR, "samples")
    HIST_CSV  = os.path.join(LOG_DIR, "history.csv")
    for d in [CKPT_DIR, WEIGHT_DIR, FIG_DIR, SAMPLE_DIR]: os.makedirs(d, exist_ok=True)
    writer = tf.summary.create_file_writer(LOG_DIR)

    def append_history_row(epoch, tr, vl, bleu, lr_p, lr_t, gstep):
        header = (not os.path.exists(HIST_CSV))
        with open(HIST_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if header: w.writerow(["epoch","train_loss","val_loss","bleu4","lr_proj","lr_t5","global_step"])
            w.writerow([epoch, f"{tr:.6f}", f"{vl:.6f}", f"{bleu:.6f}",
                        f"{lr_p:.8f}", f"{lr_t:.8f}", int(gstep)])

    history=[]; early=EarlyStoppingTF(patience=cfg.early_patience, path=os.path.join(CKPT_DIR,"best.weights.h5"))
    best_val=float("inf")

    for epoch in range(cfg.num_epochs):
        print(f"\nEpoch {epoch+1}/{cfg.num_epochs}")
        total_loss=0.0; step_in_epoch=0; micro=0
        pbar = tf.keras.utils.Progbar(steps_per_epoch, stateful_metrics=["loss","lr_proj","lr_t5"])

        for images, cap_ids in train_loader:
            if step_in_epoch >= steps_per_epoch: break
            if images.dtype != tf.float32: images = tf.cast(images, tf.float32)
            if cap_ids.dtype != tf.int32:  cap_ids= tf.cast(cap_ids, tf.int32)

            loss = train_step_acc(images, cap_ids)
            total_loss += float(loss.numpy()); micro += 1

            if micro % cfg.acc_steps == 0:
                apply_accumulated_grads()
                global_step.assign_add(1)
                step_in_epoch += 1
                lr_p = float(sched_proj(global_step).numpy())
                lr_t = float(sched_t5(global_step).numpy())
                pbar.update(step_in_epoch, values=[("loss", float(loss.numpy())),
                                                   ("lr_proj", lr_p), ("lr_t5", lr_t)])
                gc.collect()

        if int(accum_step.numpy()) > 0:
            apply_accumulated_grads()
            global_step.assign_add(1)
            step_in_epoch += 1

        avg_train = total_loss / max(step_in_epoch*cfg.acc_steps, 1)
        print(f"Epoch {epoch+1} - Avg Train Loss(per micro-step): {avg_train:.4f}")

        avg_val_loss, avg_bleu = evaluate_model_tf(
            model, val_loader, tokenizer,
            preprocess_func=preprocess_caption_for_eval,
            pad_token_id=tokenizer.pad_token_id, desc="Evaluating"
        )
        print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f} | BLEU-4: {avg_bleu:.4f}")

        with writer.as_default():
            tf.summary.scalar("train/loss", avg_train, step=int(global_step.numpy()))
            tf.summary.scalar("val/loss",   avg_val_loss, step=int(global_step.numpy()))
            tf.summary.scalar("val/BLEU4",  avg_bleu, step=int(global_step.numpy()))
            tf.summary.scalar("lr/proj",    float(sched_proj(global_step).numpy()), step=int(global_step.numpy()))
            tf.summary.scalar("lr/t5",      float(sched_t5(global_step).numpy()),   step=int(global_step.numpy()))
            writer.flush()

        append_history_row(epoch+1, avg_train, avg_val_loss, avg_bleu,
                           float(sched_proj(global_step).numpy()),
                           float(sched_t5(global_step).numpy()),
                           int(global_step.numpy()))
        history.append({"epoch":epoch+1,"train_loss":avg_train,"val_loss":avg_val_loss,"bleu4":avg_bleu})
        plot_curves(history, os.path.join(FIG_DIR, "curves.png"))

        try: save_samples(model, val_loader, tokenizer, SAMPLE_DIR, epoch+1, int(global_step.numpy()), k=log_cfg.save_max_samples)
        except Exception as e: print("[WARN] sample error:", e)
        try:
            ep_path=os.path.join(WEIGHT_DIR, f"epoch_{epoch+1:03d}.weights.h5")
            model.save_weights(ep_path)
            print(f"[Saved weights] {ep_path}")
        except Exception as e: print("[WARN] save weights error:", e)

        if avg_val_loss < best_val:
            best_val = avg_val_loss
        early(avg_val_loss, model)
        if early.early:
            print("Early stopping."); break

    print("\nHuấn luyện hoàn tất.")
    print(f"TensorBoard: {LOG_DIR}")
    print(f"Weights-by-epoch: {WEIGHT_DIR}")
    print(f"History CSV: {HIST_CSV}")
    print(f"Samples: {SAMPLE_DIR}")
