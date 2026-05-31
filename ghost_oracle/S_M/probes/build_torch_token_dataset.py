#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
BUILD TORCH TOKEN RETRIEVAL DATASET — FINAL DATASET BUILDER
===============================================================================

Purpose
-------
Final PyTorch/HuggingFace dataset builder for the Ghost Oracle Suite token
retrieval projector benchmark.

This script builds a real token-retrieval dataset from a transformer and saves
it in the exact schema expected by:

    token_retrieval_projector.py --dataset <file>.npz

This script does NOT run the projector benchmark. It only creates:

    queries      float32, shape (Nq, d)
    keys         float32, shape (Nk, d)
    true_ids     int64,   shape (Nq,)
    candidates   int64,   shape (Nq, candidate_k)

Where:
    queries[i]       = transformer hidden state for a token position
    keys[j]          = token embedding vector for a vocabulary token
    true_ids[i]      = row index in keys for the target token
    candidates[i]    = candidate key-row IDs including true_ids[i]

Default target mode
-------------------
    self_token

Meaning:
    query = hidden state at position t
    target = actual token id at position t

This tests whether the retrieval system can recover token identity from
contextual hidden states.

Other mode:
    next_token

Meaning:
    query = hidden state at position t
    target = token id at position t+1

This is harder and more LM-like, but only makes clean sense for causal models.

Usage
-----
Default small run, likely downloads model if not cached:

    python ghost_oracle/S_M/S_M_token/build_torch_token_dataset.py

Use a local/cached model only:

    python ghost_oracle/S_M/S_M_token/build_torch_token_dataset.py --local-files-only

Use your own text:

    python ghost_oracle/S_M/S_M_token/build_torch_token_dataset.py --text-file data/corpus.txt

Build next-token retrieval:

    python ghost_oracle/S_M/S_M_token/build_torch_token_dataset.py --target-mode next_token

Then benchmark:

    python ghost_oracle/S_M/S_M_token/token_retrieval_projector.py ^
      --dataset data/token_retrieval_torch_<TAG>.npz ^
      --qpu-base data/sm_data_plus_<JOB_ID>.npz

Notes
-----
- Requires torch and transformers.
- Uses model input embeddings as keys.
- Candidate sets are sampled from the token vocabulary subset used in the key
  matrix.
- If --attack is enabled, a coherent same-dimension spike is added to a fraction
  of key vectors and optionally to queries, matching the synthetic harness.
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


try:
    import torch
    _HAVE_TORCH = True
except Exception:
    torch = None
    _HAVE_TORCH = False

try:
    from transformers import AutoModel, AutoTokenizer
    _HAVE_TRANSFORMERS = True
except Exception:
    AutoModel = None
    AutoTokenizer = None
    _HAVE_TRANSFORMERS = False


# =============================================================================
# PATHS / IO
# =============================================================================

HERE = Path(__file__).resolve().parent

DATA_DIR = HERE / "data"
ANALYSIS_DIR = HERE / "analysis"


DEFAULT_TEXT = """
Ghost Oracle Suite studies projection operators across analytical, classical,
and physical quantum substrates. The important lesson is not to force the device
to be the thing we expected, but to freeze the record, build controls, and ask
what operator was actually computed.

The S_M path treats repetition-code syndrome data as a spacetime field rather
than a scalar logical-error statistic. The field has edge profiles, detection
events, temporal gradients, spatial gradients, and stress-tensor structure.

The token retrieval projector test uses a shared candidate set and compares
standard cosine retrieval against bounded projected retrieval. A coherent
same-dimension spike attack can collapse dot-product or cosine retrieval, while
a bounded projection coordinate can limit single-axis domination.

Build, break, fix, document, repeat. If a result is an optimizer, call it an
optimizer. If it is a projector ingredient, identify which coordinate it
supplies. If a field term changes ordering, measure rank delta.
"""


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_safe(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < eps] = 1.0
    return (X / n).astype(np.float32)


def read_texts(path: Optional[str], repeat_default: int) -> str:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing text file: {p}")
        return p.read_text(encoding="utf-8", errors="ignore")
    return ("\n" + DEFAULT_TEXT.strip() + "\n") * max(1, int(repeat_default))


# =============================================================================
# MODEL / TOKENIZATION
# =============================================================================

def require_deps() -> None:
    if not _HAVE_TORCH:
        raise RuntimeError("torch is not installed. Install PyTorch first.")
    if not _HAVE_TRANSFORMERS:
        raise RuntimeError("transformers is not installed. Install with: pip install transformers")


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(model_name: str, device: str, local_files_only: bool):
    require_deps()

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)

    # We manually chunk long text before sending it to the model. Some
    # tokenizers warn when the raw text exceeds model_max_length even though we
    # never feed that long sequence directly. Disable that warning where the
    # tokenizer supports it.
    try:
        tok.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
    except Exception:
        pass

    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    model.eval()
    model.to(device)

    if tok.pad_token is None:
        # Many causal models do not define pad_token. For chunked inference this
        # is fine; use EOS if available, otherwise unknown.
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.unk_token is not None:
            tok.pad_token = tok.unk_token

    return model, tok


def tokenize_to_chunks(
    tokenizer,
    text: str,
    max_length: int,
    stride: int,
    max_chunks: int,
) -> List[Dict[str, Any]]:
    enc = tokenizer(
        text,
        return_tensors=None,
        add_special_tokens=True,
        truncation=False,
    )
    ids = list(map(int, enc["input_ids"]))
    if len(ids) < 4:
        raise ValueError("Text produced too few tokens.")

    max_length = int(max_length)
    stride = int(stride)
    if max_length < 8:
        raise ValueError("--max-length should be >= 8")
    if stride <= 0:
        stride = max_length

    chunks = []
    start = 0
    while start < len(ids):
        chunk = ids[start:start + max_length]
        if len(chunk) < 4:
            break
        chunks.append({"input_ids": chunk, "start": start})
        if max_chunks > 0 and len(chunks) >= max_chunks:
            break
        if start + max_length >= len(ids):
            break
        start += stride

    return chunks


def hidden_states_for_chunks(
    model,
    tokenizer,
    chunks: List[Dict[str, Any]],
    device: str,
    layer: int,
    batch_size: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Returns lists:
        hidden chunks, token id chunks, attention mask chunks
    """
    h_list: List[np.ndarray] = []
    id_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    with torch.no_grad():
        for b0 in range(0, len(chunks), batch_size):
            batch = chunks[b0:b0 + batch_size]
            L = max(len(c["input_ids"]) for c in batch)

            input_ids = np.full((len(batch), L), int(pad_id), dtype=np.int64)
            attn = np.zeros((len(batch), L), dtype=np.int64)

            for i, c in enumerate(batch):
                ids = np.asarray(c["input_ids"], dtype=np.int64)
                input_ids[i, :len(ids)] = ids
                attn[i, :len(ids)] = 1

            ids_t = torch.as_tensor(input_ids, dtype=torch.long, device=device)
            attn_t = torch.as_tensor(attn, dtype=torch.long, device=device)

            out = model(input_ids=ids_t, attention_mask=attn_t, output_hidden_states=True)

            hs = out.hidden_states
            # Python negative layer indexing works naturally.
            H = hs[int(layer)]
            H_np = H.detach().float().cpu().numpy().astype(np.float32)

            for i, c in enumerate(batch):
                valid = int(np.sum(attn[i]))
                h_list.append(H_np[i, :valid].copy())
                id_list.append(input_ids[i, :valid].copy())
                mask_list.append(attn[i, :valid].copy())

    return h_list, id_list, mask_list


def get_input_embedding_matrix(model) -> np.ndarray:
    emb = model.get_input_embeddings()
    W = emb.weight.detach().float().cpu().numpy().astype(np.float32)
    return W


# =============================================================================
# DATASET CONSTRUCTION
# =============================================================================

def collect_query_targets(
    hidden_chunks: List[np.ndarray],
    id_chunks: List[np.ndarray],
    tokenizer,
    target_mode: str,
    skip_special: bool,
    max_queries: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        queries, target_vocab_ids, source_positions
    """
    special = set()
    if skip_special:
        special.update(int(x) for x in tokenizer.all_special_ids)

    queries = []
    targets = []
    positions = []

    for ci, (H, ids) in enumerate(zip(hidden_chunks, id_chunks)):
        ids = np.asarray(ids, dtype=np.int64)

        # self_token:
        #   Hidden state at t retrieves the token that actually appears at t.
        #   This is not language modeling, but it is a clean representation
        #   retrieval probe.
        if target_mode == "self_token":
            for t in range(len(ids)):
                tid = int(ids[t])
                if tid in special:
                    continue
                queries.append(H[t])
                targets.append(tid)
                positions.append((ci, t))
        # next_token:
        #   Hidden state at t retrieves the token that appears at t+1.
        #   This is harder and closer to LM-style prediction. It is also the
        #   mode that produced the strongest current projected-vs-cosine run.
        elif target_mode == "next_token":
            for t in range(len(ids) - 1):
                tid = int(ids[t + 1])
                if tid in special:
                    continue
                queries.append(H[t])
                targets.append(tid)
                positions.append((ci, t))
        else:
            raise ValueError(f"unknown target_mode: {target_mode}")

    if not queries:
        raise ValueError("No usable query/target pairs collected.")

    Q = np.asarray(queries, dtype=np.float32)
    T = np.asarray(targets, dtype=np.int64)
    P = np.asarray(positions, dtype=np.int64)

    rng = np.random.default_rng(seed)
    if max_queries > 0 and len(Q) > max_queries:
        idx = rng.choice(np.arange(len(Q)), size=max_queries, replace=False)
        idx.sort()
        Q = Q[idx]
        T = T[idx]
        P = P[idx]

    return Q, T, P


def choose_key_vocab(
    target_vocab_ids: np.ndarray,
    vocab_size: int,
    extra_vocab: int,
    seed: int,
    forbid: Optional[Sequence[int]] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target_unique = np.unique(target_vocab_ids.astype(np.int64))

    forbidden = set(int(x) for x in (forbid or []))
    target_unique = np.asarray([x for x in target_unique.tolist() if int(x) not in forbidden], dtype=np.int64)

    all_ids = np.arange(vocab_size, dtype=np.int64)
    if forbidden:
        mask = np.ones(vocab_size, dtype=bool)
        for x in forbidden:
            if 0 <= int(x) < vocab_size:
                mask[int(x)] = False
        all_ids = all_ids[mask]

    target_set = set(int(x) for x in target_unique.tolist())
    pool = np.asarray([x for x in all_ids.tolist() if int(x) not in target_set], dtype=np.int64)

    n_extra = min(max(0, int(extra_vocab)), len(pool))
    if n_extra:
        extra = rng.choice(pool, size=n_extra, replace=False).astype(np.int64)
        key_vocab = np.concatenate([target_unique, extra])
    else:
        key_vocab = target_unique

    key_vocab = np.unique(key_vocab.astype(np.int64))
    key_vocab.sort()
    return key_vocab


def build_candidate_sets(
    target_vocab_ids: np.ndarray,
    key_vocab_ids: np.ndarray,
    candidate_k: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Candidate IDs are row indices into the keys array, not raw vocab IDs.
    true_ids are also row indices into keys.
    """
    rng = np.random.default_rng(seed)
    vocab_to_row = {int(v): i for i, v in enumerate(key_vocab_ids.tolist())}

    true_ids = np.asarray([vocab_to_row[int(v)] for v in target_vocab_ids.tolist()], dtype=np.int64)
    n_keys = len(key_vocab_ids)
    k = min(int(candidate_k), n_keys)
    if k < 2:
        raise ValueError("candidate_k must be >= 2 after key-vocab construction.")

    all_rows = np.arange(n_keys, dtype=np.int64)
    candidates = np.empty((len(true_ids), k), dtype=np.int64)

    for i, tid in enumerate(true_ids):
        tid = int(tid)
        pool = all_rows[all_rows != tid]
        distractors = rng.choice(pool, size=k - 1, replace=False)
        row = np.concatenate([[tid], distractors]).astype(np.int64)
        rng.shuffle(row)
        candidates[i] = row

    return true_ids, candidates


def inject_attack(
    queries: np.ndarray,
    keys: np.ndarray,
    candidate_key_vocab_ids: np.ndarray,
    attack_fraction: float,
    attack_magnitude: float,
    query_attack_magnitude: float,
    attack_dim: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    Q = queries.copy().astype(np.float32)
    K = keys.copy().astype(np.float32)
    n_keys, dim = K.shape
    attack_dim = int(attack_dim) % dim

    attacked = np.zeros(n_keys, dtype=bool)
    n_attack = int(round(float(attack_fraction) * n_keys))
    n_attack = max(0, min(n_attack, n_keys))

    if n_attack > 0 and attack_magnitude != 0.0:
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(n_keys), size=n_attack, replace=False)
        attacked[idx] = True
        K[idx, attack_dim] += np.float32(attack_magnitude)

    if query_attack_magnitude != 0.0:
        Q[:, attack_dim] += np.float32(query_attack_magnitude)

    return Q, K, attacked, attack_dim


def token_text_map(tokenizer, vocab_ids: np.ndarray, max_items: int = 5000) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i, vid in enumerate(vocab_ids[:max_items].tolist()):
        try:
            out[str(int(vid))] = tokenizer.decode([int(vid)])
        except Exception:
            out[str(int(vid))] = ""
    return out


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a PyTorch/HuggingFace token retrieval dataset for the projection harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", default="distilgpt2", help="HuggingFace model name or local path.")
    p.add_argument("--local-files-only", action="store_true", help="Do not download; use cached/local model only.")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    p.add_argument("--layer", type=int, default=-1, help="Hidden-state layer index. -1 = final layer.")
    p.add_argument("--batch-size", type=int, default=4)

    p.add_argument("--text-file", default=None)
    p.add_argument("--repeat-default-text", type=int, default=80)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--max-chunks", type=int, default=0, help="0 = all chunks.")

    p.add_argument("--target-mode", choices=["self_token", "next_token"], default="self_token")
    p.add_argument("--skip-special", action="store_true", default=True)
    p.add_argument("--keep-special", action="store_true", help="Override --skip-special and keep special tokens.")

    p.add_argument("--max-queries", type=int, default=2000)
    p.add_argument("--candidate-k", type=int, default=512)
    p.add_argument("--extra-vocab", type=int, default=8192, help="Extra random vocab keys besides observed target tokens.")
    p.add_argument("--normalize", action="store_true", help="L2-normalize queries and keys before saving.")
    p.add_argument("--seed", type=int, default=20260529)

    p.add_argument("--attack", action="store_true", help="Inject coherent spike attack into saved keys/queries.")
    p.add_argument("--attack-fraction", type=float, default=0.05)
    p.add_argument("--attack-magnitude", type=float, default=16.0)
    p.add_argument("--query-attack-magnitude", type=float, default=8.0)
    p.add_argument("--attack-dim", type=int, default=0)

    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=None, help="Output .npz path. Defaults to data/token_retrieval_torch_<tag>.npz")
    p.add_argument("--meta-out", default=None, help="Optional metadata JSON path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_all = time.time()

    if args.keep_special:
        args.skip_special = False

    device = pick_device(args.device)

    tag = args.tag
    if tag is None:
        safe_model = str(args.model).replace("/", "_").replace("\\", "_").replace(":", "_")
        tag = f"{safe_model}_{args.target_mode}_{now_tag()}"

    out_path = Path(args.out) if args.out else DATA_DIR / f"token_retrieval_torch_{tag}.npz"
    meta_path = Path(args.meta_out) if args.meta_out else out_path.with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 118}")
    print("  BUILD TORCH TOKEN RETRIEVAL DATASET — FINAL DATASET BUILDER")
    print(f"{'=' * 118}")
    print(f"  Model        : {args.model}")
    print(f"  Device       : {device}")
    print(f"  Target mode  : {args.target_mode}")
    print(f"  Output       : {out_path}")
    print(f"  Metadata     : {meta_path}")

    print("\n[LOAD]")
    model, tokenizer = load_model_and_tokenizer(args.model, device, args.local_files_only)
    emb = get_input_embedding_matrix(model)
    vocab_size, emb_dim = emb.shape
    print(f"  vocab size   : {vocab_size}")
    print(f"  embed dim    : {emb_dim}")

    print("\n[TEXT]")
    text = read_texts(args.text_file, args.repeat_default_text)
    chunks = tokenize_to_chunks(
        tokenizer=tokenizer,
        text=text,
        max_length=args.max_length,
        stride=args.stride,
        max_chunks=args.max_chunks,
    )
    total_tokens = sum(len(c["input_ids"]) for c in chunks)
    print(f"  chunks       : {len(chunks)}")
    print(f"  chunk tokens : {total_tokens}")

    print("\n[MODEL]")
    H_chunks, id_chunks, mask_chunks = hidden_states_for_chunks(
        model=model,
        tokenizer=tokenizer,
        chunks=chunks,
        device=device,
        layer=args.layer,
        batch_size=args.batch_size,
    )

    print("\n[PAIRS]")
    queries, target_vocab_ids, source_positions = collect_query_targets(
        hidden_chunks=H_chunks,
        id_chunks=id_chunks,
        tokenizer=tokenizer,
        target_mode=args.target_mode,
        skip_special=args.skip_special,
        max_queries=args.max_queries,
        seed=args.seed,
    )
    print(f"  query pairs  : {queries.shape}")
    print(f"  targets uniq : {len(np.unique(target_vocab_ids))}")

    forbidden = tokenizer.all_special_ids if args.skip_special else []
    key_vocab_ids = choose_key_vocab(
        target_vocab_ids=target_vocab_ids,
        vocab_size=vocab_size,
        extra_vocab=args.extra_vocab,
        seed=args.seed + 11,
        forbid=forbidden,
    )
    keys = emb[key_vocab_ids].astype(np.float32)

    true_ids, candidates = build_candidate_sets(
        target_vocab_ids=target_vocab_ids,
        key_vocab_ids=key_vocab_ids,
        candidate_k=args.candidate_k,
        seed=args.seed + 22,
    )

    attacked_keys = np.zeros(len(keys), dtype=bool)
    attack_dim = int(args.attack_dim) % keys.shape[1]

    if args.attack:
        queries, keys, attacked_keys, attack_dim = inject_attack(
            queries=queries,
            keys=keys,
            candidate_key_vocab_ids=key_vocab_ids,
            attack_fraction=args.attack_fraction,
            attack_magnitude=args.attack_magnitude,
            query_attack_magnitude=args.query_attack_magnitude,
            attack_dim=args.attack_dim,
            seed=args.seed + 33,
        )

    if args.normalize:
        queries = l2_normalize(queries)
        keys = l2_normalize(keys)

    print("\n[DATASET]")
    print(f"  queries      : {queries.shape}")
    print(f"  keys         : {keys.shape}")
    print(f"  candidates   : {candidates.shape}")
    print(f"  attacked     : {int(attacked_keys.sum())} / {len(attacked_keys)}")
    print(f"  attack dim   : {attack_dim}")

    # Raw vocab IDs are saved as metadata arrays for interpretation/debugging.
    np.savez_compressed(
        out_path,
        queries=queries.astype(np.float32),
        keys=keys.astype(np.float32),
        true_ids=true_ids.astype(np.int64),
        candidates=candidates.astype(np.int64),
        attacked_keys=attacked_keys.astype(bool),
        attack_dim=np.asarray(attack_dim, dtype=np.int64),

        # Extra interpretability arrays.
        target_vocab_ids=target_vocab_ids.astype(np.int64),
        key_vocab_ids=key_vocab_ids.astype(np.int64),
        source_positions=source_positions.astype(np.int64),
    )

    meta = {
        "schema": "token_retrieval_torch",
        "model": args.model,
        "device": device,
        "layer": args.layer,
        "target_mode": args.target_mode,
        "text_file": args.text_file,
        "repeat_default_text": args.repeat_default_text,
        "max_length": args.max_length,
        "stride": args.stride,
        "max_chunks": args.max_chunks,
        "max_queries": args.max_queries,
        "candidate_k": args.candidate_k,
        "extra_vocab": args.extra_vocab,
        "normalize": args.normalize,
        "skip_special": args.skip_special,
        "attack": args.attack,
        "attack_fraction": args.attack_fraction,
        "attack_magnitude": args.attack_magnitude,
        "query_attack_magnitude": args.query_attack_magnitude,
        "attack_dim": attack_dim,
        "queries_shape": list(queries.shape),
        "keys_shape": list(keys.shape),
        "candidates_shape": list(candidates.shape),
        "unique_targets": int(len(np.unique(target_vocab_ids))),
        "attacked_keys": int(attacked_keys.sum()),
        "token_text_sample": token_text_map(tokenizer, key_vocab_ids, max_items=5000),
        "seconds": time.time() - t_all,
        "next_step": (
            "Run token_retrieval_projector.py --dataset "
            f"{out_path} --qpu-base data/sm_data_plus_<JOB_ID>.npz"
        ),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(meta), f, indent=2)

    print(f"\n[SAVED] {out_path}")
    print(f"[META]  {meta_path}")
    print("\n[NEXT]")
    print(f"  python ghost_oracle/S_M/S_M_token/token_retrieval_projector.py --dataset {out_path} --qpu-base data/sm_data_plus_<JOB_ID>.npz")
    print(f"{'=' * 118}\n")


if __name__ == "__main__":
    main()
