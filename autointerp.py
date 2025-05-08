from utils import *
from openai import OpenAI
import os
import json
import torch
from typing_extensions import Literal

# Autointerp functions
def create_prompt(
    data,
    buffer: int = 10,
) -> dict[Literal["system", "user", "assistant"], str]:
    """
    Returns the system, user & assistant prompts for autointerp.
    """
    
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
    data,
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

    prompts = create_prompt(data, buffer)

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