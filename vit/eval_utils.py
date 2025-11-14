import re
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import T5Tokenizer

def preprocess_caption_for_eval(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _pad_to_tensor_first_caption(captions_token_ids_lists, pad_token_id: int, dtype=np.int32):
    first_caps = []
    for caps_list in captions_token_ids_lists:
        cap0 = caps_list[0]
        if isinstance(cap0, tf.Tensor):
            cap0 = cap0.numpy().tolist()
        first_caps.append(list(map(int, cap0)))
    max_len = max(len(seq) for seq in first_caps) if first_caps else 0
    if max_len == 0:
        return None
    arr = np.full((len(first_caps), max_len), pad_token_id, dtype=dtype)
    for i, seq in enumerate(first_caps):
        L = len(seq)
        arr[i, :L] = np.asarray(seq, dtype=dtype)
    return arr

def calculate_bleu_score_tf(pred_ids_batch, references_text_lists, tokenizer: T5Tokenizer, preprocess_func):
    smoothie = SmoothingFunction().method4
    if isinstance(pred_ids_batch, tf.Tensor):
        pred_ids_batch = pred_ids_batch.numpy().tolist()
    pred_texts = tokenizer.batch_decode(pred_ids_batch, skip_special_tokens=True)
    pred_texts = [preprocess_func(s) for s in pred_texts]

    total_bleu, count = 0.0, 0
    for pred_sent, refs in zip(pred_texts, references_text_lists):
        processed_refs = [preprocess_func(r).split() for r in refs]
        pred_words = pred_sent.split()
        if pred_words and any(len(r) > 0 for r in processed_refs):
            bleu = sentence_bleu(
                processed_refs, pred_words,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothie,
            )
            total_bleu += bleu
            count += 1
    return (total_bleu / count) if count > 0 else 0.0

def evaluate_model_tf(model,
                      val_dataset,
                      tokenizer: T5Tokenizer,
                      preprocess_func=preprocess_caption_for_eval,
                      pad_token_id=0,
                      desc='Evaluating'):
    total_val_loss = 0.0
    total_bleu = 0.0
    num_batches = 0

    for images, captions_token_ids_lists in tqdm(val_dataset, desc=desc):
        single_caps = _pad_to_tensor_first_caption(captions_token_ids_lists, pad_token_id, dtype=np.int32)
        if single_caps is not None:
            single_caps_tf = tf.convert_to_tensor(single_caps, dtype=tf.int32)
            val_loss = model(images=images, captions=single_caps_tf, training=False, return_loss=True)
            val_loss = float(val_loss.numpy() if isinstance(val_loss, tf.Tensor) else val_loss)
        else:
            val_loss = 0.0

        generated_ids = model(images=images, training=False)
        references_text_lists = []
        for caps_token_ids_list in captions_token_ids_lists:
            ref_texts = []
            for cap in caps_token_ids_list:
                if isinstance(cap, tf.Tensor):
                    cap = cap.numpy().tolist()
                ref_texts.append(tokenizer.decode(cap, skip_special_tokens=True))
            references_text_lists.append(ref_texts)

        batch_bleu = calculate_bleu_score_tf(generated_ids, references_text_lists, tokenizer, preprocess_func)
        total_val_loss += val_loss
        total_bleu += batch_bleu
        num_batches += 1

    avg_val_loss = (total_val_loss / num_batches) if num_batches > 0 else 0.0
    avg_bleu = (total_bleu / num_batches) if num_batches > 0 else 0.0
    return avg_val_loss, avg_bleu
