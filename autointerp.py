import os
import json
from typing import Literal, List, Tuple

import einops
import torch
from jaxtyping import Float, Int
from openai import OpenAI
from rich import print as rprint
from rich.table import Table
from sae_lens import (
    SAE,
    ActivationsStore,
    HookedSAETransformer,
)
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens.utils import to_numpy


# Helper functions (originally from the notebook)
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


def index_with_buffer(
    x: Float[Tensor, "batch seq"], indices: Int[Tensor, "k 2"], buffer: int | None = None
) -> Float[Tensor, "k *buffer_x2_plus1"]:
    """
    Indexes into `x` with `indices` (which should have come from the `get_k_largest_indices` function), and takes a
    +-buffer range around each indexed element. If `indices` are less than `buffer` away from the start of a sequence
    then we just take the first `2*buffer+1` elems (same for at the end of a sequence).

    If `buffer` is None, then we don't add any buffer and just return the elements at the given indices.
    """
    rows, cols = indices.unbind(dim=-1)
    if buffer is not None:
        rows = einops.repeat(rows, "k -> k buffer", buffer=buffer * 2 + 1)
        # Clamp columns to be within bounds after adding buffer range
        cols_clamped = torch.clamp(cols, min=buffer, max=x.size(1) - buffer - 1)
        cols_buffered = einops.repeat(cols_clamped, "k -> k buffer", buffer=buffer * 2 + 1) + torch.arange(
            -buffer, buffer + 1, device=cols.device
        )
        # Ensure final indices are within valid range
        cols_final = torch.clamp(cols_buffered, min=0, max=x.size(1) - 1)

    else:
        cols_final = cols

    return x[rows, cols_final]


def display_top_seqs(data: list[tuple[float, list[str], int]]):
    """
    Given a list of (activation: float, str_toks: list[str], seq_pos: int), displays a table of these sequences, with
    the relevant token highlighted.

    We also turn newlines into "\\n", and remove unknown tokens � (usually weird quotation marks) for readability.
    """
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
) -> list[tuple[float, list[str], int]]:
    """
    Returns the max activating examples across a number of batches from the activations store.
    """
    sae_acts_post_hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"

    # Create list to store the top k activations for each batch. Once we're done,
    # we'll filter this to only contain the top k over all batches
    data = []

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

    return sorted(data, key=lambda x: x[0], reverse=True)[:k]


# Autointerp functions
def create_prompt(
    model: HookedSAETransformer,
    sae: SAE,
    act_store: ActivationsStore,
    latent_idx: int,
    total_batches: int = 100,
    k: int = 15,
    buffer: int = 10,
) -> dict[Literal["system", "user", "assistant"], str]:
    """
    Returns the system, user & assistant prompts for autointerp.
    """

    data = fetch_max_activating_examples(model, sae, act_store, latent_idx, total_batches, k, buffer)
    str_formatted_examples = "\n".join(
        f"{i+1}. {''.join(f'<<{tok}>>' if j == buffer else tok for j, tok in enumerate(seq[1]))}"
        for i, seq in enumerate(data)
    )
    return {
        "system": "We're studying neurons in a neural network. Each neuron activates on some particular word or concept in a short document. The activating words in each document are indicated with << ... >>. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is activating on. Try to be specific in your explanations, although don't be so specific that you exclude some of the examples from matching your explanation. Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant. Keep the explanation as short and simple as possible, limited to 20 words or less. Omit punctuation and formatting. You should avoid giving long lists of words.",
        "user": f"""The activating documents are given below:\n\n{str_formatted_examples}""",
        "assistant": "this neuron fires on",
    }


def get_autointerp_explanation(
    model: HookedSAETransformer,
    sae: SAE,
    act_store: ActivationsStore,
    latent_idx: int,
    total_batches: int = 100,
    k: int = 15,
    buffer: int = 10,
    n_completions: int = 1,
) -> list[str]:
    """
    Queries OpenAI's API using prompts returned from `create_prompt`, and returns
    a list of the completions.
    """
    # Ensure the API key is set in your environment variables
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        raise ValueError("GENAI_API_KEY environment variable not set.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/" # Using Google's endpoint as per notebook
    )

    prompts = create_prompt(model, sae, act_store, latent_idx, total_batches, k, buffer)

    try:
        result = client.chat.completions.create(
            model="gemini-2.0-flash", # Using Gemini model as per notebook
            messages=[
                {"role": "system", "content": prompts["system"]},
                {"role": "user", "content": prompts["user"]},
                {"role": "assistant", "content": prompts["assistant"]},
            ],
            n=n_completions,
            max_tokens=50,
            stream=False,
        )
        return [choice.message.content for choice in result.choices]
    except Exception as e:
        print(f"Error calling AI API: {e}")
        return []


if __name__ == "__main__":
    # Setup (similar to the notebook)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load SAE
    try:
        sae = SAE.from_pretrained(
            release="anhtu77/sae-tiny-stories-1L-21M",
            sae_id="sae_ex32",
            device=device,
        )[0]
        print("SAE loaded successfully.")
    except Exception as e:
        print(f"Error loading SAE: {e}")
        exit()

    # Load the base model
    try:
        model = HookedSAETransformer.from_pretrained(
            "tiny-stories-1L-21M",
            device=device,
        )
        print("Base model loaded successfully.")
    except Exception as e:
        print(f"Error loading base model: {e}")
        exit()

    # Instantiate ActivationStore
    try:
        activation_store = ActivationsStore.from_sae(
            model=model,
            sae=sae,
            streaming=True,
            store_batch_size_prompts=2,
            train_batch_size_tokens=4096,
            n_batches_in_buffer=32,
            device=device,
        )
        print("ActivationStore initialized.")
    except Exception as e:
        print(f"Error initializing ActivationStore: {e}")
        exit()

    total_latents = 1024*32
    num_batches_for_examples = 100  
    num_examples_for_prompt = 15   
    example_buffer = 10              
    num_completions = 1               

    all_explanations = {}

    print(f"\nGenerating autointerp explanations for all latents (0 to {total_latents - 1}) ...")
    for latent_idx in range(total_latents):
        print(f"\nProcessing latent index {latent_idx} ...")
        completions = get_autointerp_explanation(
            model=model,
            sae=sae,
            act_store=activation_store,
            latent_idx=latent_idx,
            total_batches=num_batches_for_examples,
            k=num_examples_for_prompt,
            buffer=example_buffer,
            n_completions=num_completions
        )
        if completions:
            explanation = completions[0]
            all_explanations[latent_idx] = explanation
            print(f"Latent {latent_idx}: {explanation!r}")
        else:
            print(f"Failed to generate explanation for latent {latent_idx}")

    # Save explanations to a JSON file in a pandas-friendly format (a list of objects)
    output_path = r"latent_data.json"
    try:
        data_to_save = [
            {"latent": latent_idx, "explanation": explanation}
            for latent_idx, explanation in sorted(all_explanations.items())
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        print(f"\nAutointerp explanations saved to {output_path}")
    except Exception as e:
        print(f"Error saving autointerp explanations: {e}")