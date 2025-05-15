from utils import *
from tabulate import tabulate
from autointerp import get_autointerp_explanation
def load_stuffs(model_name: str, sae_id: str):
    """
    Inputs: model_name and sae_id.
    returns: model, sae, activation_store, and projection_onto_unembed.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedSAETransformer.from_pretrained_no_processing(model_name, device=device)
    # Adjust release as needed; here we assume "anhtu77/sae-{model_name}"
    if model_name == "tiny-stories-1L-21M":
        release_name = f"anhtu77/sae-{model_name}"
    else:
        release_name = f"anhtu77/sae-topk-32-{model_name}"

    sae = SAE.from_pretrained(
        release=release_name,
        sae_id=sae_id,
        device=device,
    )[0]

    activation_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        streaming=True,
        store_batch_size_prompts=2,
        train_batch_size_tokens=4096,
        n_batches_in_buffer=32,
        device=device,
    )

    projection_onto_unembed = sae.W_dec @ model.W_U

    return model, sae, activation_store, projection_onto_unembed

def get_dashboard_data(model, sae, act_store, projection_onto_unembed, prompt: str) -> dict:
    """
    Fetches the top activations for the given prompt, the autointerp explanation, and the top and bottom logits for the latent.
    Returns a dictionary with keys:
      - "top_latents": list of latent indices (as ints),
      - "max_examples": list of max activating examples (per latent),
      - "top_logits": list of top tokens (per latent),
      - "bottom_logits": list of bottom tokens (per latent),
      - "top_logits_values": list of top token logit values (per latent),
      - "bottom_logits_values": list of bottom token logit values (per latent).
    """
    tokenized_prompt = model.to_str_tokens(prompt)
    print("Tokenized prompt:")
    print(" ".join(f"{i} {token}" for i, token in enumerate(tokenized_prompt)))
    
    # Run model to get SAE activations
    _, cache = model.run_with_cache_with_saes(
        prompt,
        saes=[sae],
        stop_at_layer=sae.cfg.hook_layer + 1,
    )
    sae_acts_post = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"][0, -1, :]  # shape: (seq_length, num_latents)
    
    # Get top 3 latent indices (topk is applied across the token-sequence dimension)
    top_latents_tensor = torch.topk(sae_acts_post, 3).indices
    top_latents = top_latents_tensor.tolist()
    
    max_examples = []
    top_logits_list = []
    bottom_logits_list = []
    top_logits_values = []
    bottom_logits_values = []
    
    for latent_idx in top_latents:
        print("Latent index:", latent_idx)
        # Get the examples that maximally activate this latent feature
        examples = fetch_max_activating_examples(
            model=model,
            sae=sae,
            act_store=act_store,
            latent_idx=torch.tensor(latent_idx),
            display=False,  # Do not print here so that the dashboard remains clean.
        )
        max_examples.append(examples)
        # Print the autointerp explanation for debugging/logging purposes.
        explanation = get_autointerp_explanation(data=examples)[0]
        print("Autointerp explanation:", explanation)
        
        # Get the top 10 (and bottom 10) tokens and their logit values associated with this latent via the projection.
        pos_logits, pos_token_ids = projection_onto_unembed[latent_idx].topk(10)
        neg_logits, neg_token_ids = projection_onto_unembed[latent_idx].topk(10, largest=False)
        pos_tokens = model.to_str_tokens(pos_token_ids)
        neg_tokens = model.to_str_tokens(neg_token_ids)
        
        top_logits_list.append(pos_tokens)
        bottom_logits_list.append(neg_tokens)
        top_logits_values.append(pos_logits.tolist())
        bottom_logits_values.append(neg_logits.tolist())
    
    return {
        "top_latents": top_latents,
        "max_examples": max_examples,
        "top_logits": top_logits_list,
        "bottom_logits": bottom_logits_list,
        "top_logits_values": top_logits_values,
        "bottom_logits_values": bottom_logits_values,
    }

def display_dashboard(model, sae, act_store, projection_onto_unembed, prompt: str) -> None:
    tokenized_prompt = model.to_str_tokens(prompt)
    print("Tokenized prompt: ")
    print(" ".join(f"{id} {token}" for id, token in enumerate(tokenized_prompt)))
    # token_id = int(input("Enter token id to visualize: "))
    _, cache = model.run_with_cache_with_saes(
        prompt, 
        saes=[sae], 
        stop_at_layer=sae.cfg.hook_layer + 1,
    )
    sae_acts_post = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"][0, -1, :]
    top_latents = torch.topk(sae_acts_post, 3).indices
    

    for i, latent_idx in enumerate(top_latents):
        print("Latent index: ", latent_idx.item())
        data = fetch_max_activating_examples(
            model=model,
            sae=sae,
            act_store=act_store,
            latent_idx=latent_idx,
            display=True,
        )
        print("Autointerp explanation: ",get_autointerp_explanation(data = data)[0])
        pos_logits, pos_token_ids = projection_onto_unembed[latent_idx].topk(10)
        pos_tokens = model.to_str_tokens(pos_token_ids)
        neg_logits, neg_token_ids = projection_onto_unembed[latent_idx].topk(10, largest=False)
        neg_tokens = model.to_str_tokens(neg_token_ids)

        print(
            tabulate(
                zip(map(repr, neg_tokens), neg_logits, map(repr, pos_tokens), pos_logits),
                headers=["Bottom tokens", "Value", "Top tokens", "Value"],
                tablefmt="simple_outline",
                stralign="right",
                numalign="left",
                floatfmt="+.3f",
            )
        )
    

if __name__ == "__main__":
    # Load model and SAE
    # model_name = "tiny-stories-1L-21M"  # or "gpt2-small"
    model_name = "gpt2-small"  # Adjust as needed
    # sae_id = "sae_ex32"  # Adjust as needed
    sae_id = "blocks.7.hook_mlp_out"  # Adjust as needed
    loaded_model, loaded_sae, loaed_act_store, loaded_proj = load_stuffs(model_name, sae_id)
    
    # Display dashboard with a prompt
    while(1):
        prompt = input("Enter a prompt: ")
        display_dashboard(loaded_model, loaded_sae, loaed_act_store, loaded_proj, prompt)
