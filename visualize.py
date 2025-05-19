from utils import *
from tabulate import tabulate
from autointerp import get_autointerp_explanation
import warnings
import argparse
warnings.filterwarnings("ignore")

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

def display_dashboard(model, sae, act_store, projection_onto_unembed, prompt: str) -> None:
    tokenized_prompt = model.to_str_tokens(prompt)
    
    # Create a table with two rows: one for indices and one for tokens
    table_data = [
        [str(i) for i in range(len(tokenized_prompt))],
        tokenized_prompt
    ]
    
    # Print the table using tabulate with a grid format and centered text
    print("Tokenized prompt:")
    print(tabulate(
        table_data,
        tablefmt="grid",
        stralign="center"
    ))
    
    print()  # for spacing
    
    token_id = int(input("Enter token id to visualize: "))
    print("Finding features activated for token", tokenized_prompt[token_id])

    _, cache = model.run_with_cache_with_saes(
        prompt, 
        saes=[sae], 
        stop_at_layer=sae.cfg.hook_layer + 1,
    )
    sae_acts_post = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"][0, token_id, :]
    top_latents = torch.topk(sae_acts_post, 3).indices

    for i, latent_idx in enumerate(top_latents):
        print("Latent index: ", latent_idx.item())
        data = fetch_max_activating_examples(
            model=model,
            sae=sae,
            act_store=act_store,
            latent_idx=latent_idx,
            k=15,
            display=True,
        )
        print("Autointerp explanation: ", get_autointerp_explanation(data=data)[0])
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
    parser = argparse.ArgumentParser(description="Visualize SAE features for a given model and SAE id.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model (e.g., 'gpt2-small')")
    parser.add_argument("--sae_id", type=str, required=True, help="SAE id (e.g., 'blocks.0.hook_mlp_out')")
    args = parser.parse_args()

    loaded_model, loaded_sae, loaed_act_store, loaded_proj = load_stuffs(args.model_name, args.sae_id)
    
    # Display dashboard with a prompt
    while True:
        prompt = input("Enter a prompt: ")
        if prompt.lower() == "exit()":
            break
        display_dashboard(loaded_model, loaded_sae, loaed_act_store, loaded_proj, prompt)
