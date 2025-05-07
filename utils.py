import torch
import pandas as pd
import einops
from jaxtyping import Float, Int
from torch import Tensor
from rich import print as rprint
from rich.table import Table
from sae_lens import (
    SAE,
    ActivationsStore,
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    SAEConfig,
    SAETrainingRunner,
    upload_saes_to_huggingface,
)
from tqdm import tqdm

def list_flatten(nested_list):
    return [x for y in nested_list for x in y]


# A very handy function Neel wrote to get context around a feature activation
def make_token_df(tokens, len_prefix=5, len_suffix=3, model=None):
    str_tokens = [model.to_str_tokens(t) for t in tokens]
    unique_token = [
        [f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens
    ]

    context = []
    prompt = []
    pos = []
    label = []
    for b in range(tokens.shape[0]):
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p - len_prefix) : p])
            if p == tokens.shape[1] - 1:
                suffix = ""
            else:
                suffix = "".join(
                    str_tokens[b][p + 1 : min(tokens.shape[1] - 1, p + 1 + len_suffix)]
                )
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            prompt.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")
    # print(len(batch), len(pos), len(context), len(label))
    return pd.DataFrame(
        dict(
            str_tokens=list_flatten(str_tokens),
            unique_token=list_flatten(unique_token),
            context=context,
            prompt=prompt,
            pos=pos,
            label=label,
        )
    )

def get_k_largest_indices(x: Float[Tensor, "batch seq"], k: int, buffer: int = 0) -> Int[Tensor, "k 2"]:
    """
    The indices of the top k elements in the input tensor, i.e. output[i, :] is the (batch, seqpos) value of the i-th
    largest element in x.

    Won't choose any elements within `buffer` from the start or end of their sequence.
    """
    if buffer > 0:
        x = x[:, buffer:-buffer]
    indices = x.flatten().topk(k=k).indices
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer
    return torch.stack((rows, cols), dim=1)

def get_k_largest_indices(
    x: Float[Tensor, "batch seq"], k: int, buffer: int = 0, no_overlap: bool = True
) -> Int[Tensor, "k 2"]:
    if buffer > 0:
        x = x[:, buffer:-buffer]
    indices = x.flatten().argsort(-1, descending=True)
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer

    if no_overlap:
        unique_indices = torch.empty((0, 2), device=x.device).long()
        while len(unique_indices) < k:
            unique_indices = torch.cat((unique_indices, torch.tensor([[rows[0], cols[0]]], device=x.device)))
            is_overlapping_mask = (rows == rows[0]) & ((cols - cols[0]).abs() <= buffer)
            rows = rows[~is_overlapping_mask]
            cols = cols[~is_overlapping_mask]
        return unique_indices

    return torch.stack((rows, cols), dim=1)[:k]


def index_with_buffer(
    x: Float[Tensor, "batch seq"], indices: Int[Tensor, "k 2"], buffer: int | None = None
) -> Float[Tensor, "k *buffer_x2_plus1"]:
    rows, cols = indices.unbind(dim=-1)
    if buffer is not None:
        rows = einops.repeat(rows, "k -> k buffer", buffer=buffer * 2 + 1)
        cols[cols < buffer] = buffer
        cols[cols > x.size(1) - buffer - 1] = x.size(1) - buffer - 1
        cols = einops.repeat(cols, "k -> k buffer", buffer=buffer * 2 + 1) + torch.arange(
            -buffer, buffer + 1, device=x.device
        )
    return x[rows, cols]


def display_top_seqs(data: list[tuple[float, list[str], int]]):
    table = Table("Act", "Sequence", title="Max Activating Examples", show_lines=True)
    for act, str_toks, seq_pos in data:
        formatted_seq = (
            "".join([f"[b u green]{str_tok}[/]" if i == seq_pos else str_tok for i, str_tok in enumerate(str_toks)])
            .replace("�", "")
            .replace("\n", "↵")
        )
        table.add_row(f"{act:.3f}", repr(formatted_seq))
    rprint(table)


def fetch_max_activating_examples(
    model: HookedSAETransformer,
    sae: SAE,
    act_store: ActivationsStore,
    latent_idx: int,
    total_batches: int = 100,
    k: int = 10,
    buffer: int = 10,
    display: bool = False,
) -> list[tuple[float, list[str], int]]:
    data = []
    sae_acts_post_hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"
    for _ in tqdm(range(total_batches), desc="Computing activations for max activating examples"):
        tokens = act_store.get_batch_tokens()
        _, cache = model.run_with_cache_with_saes(
            tokens,
            saes=[sae],
            stop_at_layer=sae.cfg.hook_layer + 1,
            names_filter=[sae_acts_post_hook_name],
        )
        acts = cache[sae_acts_post_hook_name][..., latent_idx]

        # Get largest indices, get the corresponding max acts, and get the surrounding indices
        k_largest_indices = get_k_largest_indices(acts, k=k, buffer=buffer)
        tokens_with_buffer = index_with_buffer(tokens, k_largest_indices, buffer=buffer)
        str_toks = [model.to_str_tokens(toks) for toks in tokens_with_buffer]
        top_acts = index_with_buffer(acts, k_largest_indices).tolist()
        data.extend(list(zip(top_acts, str_toks, [buffer] * len(str_toks))))

    data = sorted(data, key=lambda x: x[0], reverse=True)[:k]
    if display:
        display_top_seqs(data)
    return data