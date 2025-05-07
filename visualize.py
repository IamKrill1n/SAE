from utils import *

def load_model_and_sae(model_name: str, sae_id: str):
    """
    Loads the model and SAE with the given names.
    Returns (model, sae, device).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedSAETransformer.from_pretrained(model_name, device=device)
    # Adjust release as needed; here we assume "anhtu77/sae-{model_name}"
    sae = SAE.from_pretrained(
        release=f"anhtu77/sae-{model_name}",
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
    return model, sae, activation_store, device

def get_dashboard_data(model, sae, activation_store, device, prompt: str) -> dict:
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
    sae_acts_post = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"][0, -1, :]
    top_latents = torch.topk(sae_acts_post, 3).indices
    
    # Fetch max activating examples for the first latent
    max_examples = fetch_max_activating_examples(
        model=model,
        sae=sae,
        act_store=activation_store,
        latent_idx=top_latents[0],
        display=False,
    )
    
    # Get top logits for the top latents
    projection_onto_unembed = sae.W_dec @ model.W_U
    _, topk_tokens = torch.topk(projection_onto_unembed[top_latents], 10, dim=1)
    top_logits = [model.to_str_tokens(token) for token in topk_tokens[0]]
    
    return {
        "top_latents": top_latents.tolist(),
        "max_examples": max_examples,
        "top_logits": top_logits,
    }

def display_dashboard(model, sae, act_store, prompt: str) -> None:
    """
    Existing function for direct command line interactive display.
    """
    # Existing code here…
    _, cache = model.run_with_cache_with_saes(
        prompt, 
        saes=[sae], 
        stop_at_layer=sae.cfg.hook_layer + 1,
    )
    sae_acts_post = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"][0, -1, :]
    top_latents = torch.topk(sae_acts_post, 3).indices
    print(top_latents)
    fetch_max_activating_examples(
        model=model,
        sae=sae,
        act_store=act_store,
        latent_idx=top_latents[0],
        display=True,
    )
    projection_onto_unembed = sae.W_dec @ model.W_U
    _, topk_tokens = torch.topk(projection_onto_unembed[top_latents], 10, dim=1)
    print("Top logits:")
    for i, token in enumerate(topk_tokens[0]):
        print(f"{i}: {model.to_str_tokens(token)}")

if __name__ == "__main__":
    # Load model and SAE
    model_name = "tiny-stories-1L-21M"  # or "gpt2-small"
    sae_id = "sae_ex32"  # Adjust as needed
    loaded_model, loaded_sae, loaed_act_store, loaded_device = load_model_and_sae(model_name, sae_id)
    
    # Display dashboard with a prompt
    prompt = "Once upon a time, there was a beautiful princess"  # input("Enter a prompt: ")
    display_dashboard(loaded_model, loaded_sae, loaed_act_store, prompt)
