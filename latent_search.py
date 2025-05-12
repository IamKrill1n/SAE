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

def get_dashboard_data(model, sae, activation_store, projection_onto_unembed, prompt: str) -> dict:
    """
    Processes the prompt to return dashboard data:
    top latents, maximum activating examples, and top logits.
    """
    
    # Get top activating latents for the prompt
    _, cache = model.run_with_cache_with_saes(
        prompt, 
        saes=[sae],
        stop_at_layer=sae.cfg.hook_layer + 1,
    )
    sae_acts_post = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"][0, -1, :] # Get the last token's activations
    top_latents = torch.topk(sae_acts_post, 3).indices # Get top 3 latents
    
    # Fetch max activating examples for all latents
    max_examples_list = []
    for latent_idx in top_latents:
        max_examples = fetch_max_activating_examples(
            model=model,
            sae=sae,
            act_store=activation_store,
            latent_idx=latent_idx,
            display=False,
        )
        max_examples_list.append(max_examples)
    # Get top logits for the top latents
    _, topk_tokens = torch.topk(projection_onto_unembed[top_latents], 10, dim=1)
    top_logits = [model.to_str_tokens(token) for i in range(len(topk_tokens)) for token in topk_tokens[i]]

    return {
        "top_latents": top_latents.tolist(),
        "max_examples": max_examples_list,
        "top_logits": top_logits,
    }

def display_dashboard(model, sae, act_store, projection_onto_unembed, latent_idx) -> None:
    
    print("Latent index: ", latent_idx)
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
        latent_idx = int(input("Enter latent index to visualize: "))
        display_dashboard(loaded_model, loaded_sae, loaed_act_store, loaded_proj, latent_idx)