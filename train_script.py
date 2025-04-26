import torch
import os
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner, upload_saes_to_huggingface

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(layer):
    total_training_steps = 30_000
    batch_size = 4096
    total_training_tokens = total_training_steps * batch_size

    lr_warm_up_steps = 0
    lr_decay_steps = total_training_steps // 5
    l1_warm_up_steps = total_training_steps // 20

    cfg = LanguageModelSAERunnerConfig(
        model_name="gpt2-small",
        hook_name="blocks." + str(layer) + ".hook_mlp_out",
        hook_layer=layer,
        d_in=768,
        dataset_path="Skylion007/openwebtext",
        is_dataset_tokenized=True,
        streaming=True,
        mse_loss_normalization=None,
        expansion_factor=32,
        b_dec_init_method="zeros",
        apply_b_dec_to_input=False,
        normalize_sae_decoder=False,
        scale_sparsity_penalty_by_decoder_norm=True,
        decoder_heuristic_init=True,
        init_encoder_as_decoder_transpose=True,
        normalize_activations="expected_average_only_in",
        lr=5e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_name="constant",
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_decay_steps,
        l1_coefficient=5,
        l1_warm_up_steps=l1_warm_up_steps,
        lp_norm=1.0,
        train_batch_size_tokens=batch_size,
        context_size=512,
        n_batches_in_buffer=64,
        training_tokens=total_training_tokens,
        store_batch_size_prompts=16,
        use_ghost_grads=False,
        feature_sampling_window=1000,
        dead_feature_window=1000,
        dead_feature_threshold=1e-4,
        log_to_wandb=True,
        wandb_project="sae_lens_tutorial",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
        device=device,
        seed=42,
        n_checkpoints=1,
        checkpoint_path="checkpoints",
        dtype="float32",
    )

    # Start training
    sparse_autoencoder = SAETrainingRunner(cfg).run()
    path = os.path.join(cfg.checkpoint_path, os.listdir(cfg.checkpoint_path)[0])
    sae_dict = {
        cfg.hook_name: path
    }
    upload_saes_to_huggingface(
        saes_dict=sae_dict,
        hf_repo_id="anhtu77/sae-gpt2-small",
    )
    print(f"Uploaded SAE for layer {layer} to Hugging Face Hub.")
    del sparse_autoencoder

def main():
    # Train the model with different layers
    for layer in range(12):
        print(f"Training layer {layer}...")
        train(layer)
        print(f"Finished training layer {layer}.")

main()