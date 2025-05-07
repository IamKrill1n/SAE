# %%
import wandb
import pandas as pd

api = wandb.Api()

project = 'tiny_stories_saes'
entity = 'hustcollab'

runs = api.runs(f"{entity}/{project}")
data = []

for run in runs:
    expansion_factor = run.config.get("expansion_factor", None)
    l1_coef = run.config.get("l1_coefficient", None)
    final_l0_norm = run.summary.get("metrics/l0", None)
    final_l2_loss = run.summary.get("losses/mse_loss", None)
    final_ce_loss_score = run.summary.get("model_performance_preservation/ce_loss_score", None)
    data.append({
        'expansion_factor': expansion_factor,
        'l1_coef': l1_coef,
        "l0_norm": final_l0_norm,
        "mse": final_l2_loss,
        "ce_loss_score": final_ce_loss_score
    })

df = pd.DataFrame(data)
print(df)

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
# Plot 1: MSE vs L0 norm
axs[0, 0].plot(df['l0_norm'], df['mse'], marker='o', linestyle='-', color='b')
axs[0, 0].set_title('MSE vs L0 Norm')
axs[0, 0].set_xlabel('L0 Norm')
axs[0, 0].set_ylabel('MSE')
axs[0, 0].grid(True)

# Plot 2: CE loss score vs L0 norm]
axs[0, 1].plot(df['l0_norm'], df['ce_loss_score'], marker='o', linestyle='-', color='r')
axs[0, 1].set_title('CE loss score vs L0 Norm')
axs[0, 1].set_xlabel('L0 Norm')
axs[0, 1].set_ylabel('CE loss score')
axs[0, 1].grid(True)

# Plot 3: MSE vs CE degradation

# Plot 1: Normalized MSE vs Dictionary size (k=32)
# for sae_type in ['batchtopk', 'topk']:
#     data = df[(df['config_sae_type'] == sae_type) & (df['k'] == 32.)]
#     data = data.sort_values(by='dictionary_size')
#     axs[0, 0].plot(data['dictionary_size'], data['normalized_mse'], 
#                    marker='o', linestyle='--', label=f"{sae_type} (k=32)")

# axs[0, 0].set_title('Normalized MSE vs Dictionary size (k=32)')
# axs[0, 0].set_xlabel('Dictionary size')
# axs[0, 0].set_ylabel('Normalized MSE')
# axs[0, 0].set_xscale('log')
# axs[0, 0].legend()
# axs[0, 0].grid(True)

# Plot 2: Normalized MSE vs k (Dict size = 12288)
# for sae_type in ['batchtopk', 'topk', 'jumprelu']:
#     data = df[(df['config_sae_type'] == sae_type) & (df['dictionary_size'] == 12288)]
#     data = data.sort_values(by='dictionary_size')
#     axs[0, 1].plot(data['l0_norm'], data['normalized_mse'], 
#                    marker='o', linestyle='--', label=sae_type)

# axs[0, 1].set_title('Normalized MSE vs k (Dict size = 12288)')
# axs[0, 1].set_xlabel('k')
# axs[0, 1].set_ylabel('Normalized MSE')
# axs[0, 1].legend()
# axs[0, 1].grid(True)

# Plot 3: CE degradation vs Dictionary size (k=32)
# for sae_type in ['batchtopk', 'topk']:
#     data = df[(df['config_sae_type'] == sae_type) & (df['k'] == 32)]
#     axs[1, 0].plot(data['dictionary_size'], data['ce_degradation'], 
#                    marker='o', linestyle='--', label=f"{sae_type} (k=32)")

# axs[1, 0].set_title('CE degradation vs Dictionary size (k=32)')
# axs[1, 0].set_xlabel('Dictionary size')
# axs[1, 0].set_ylabel('CE degradation')
# axs[1, 0].set_xscale('log')
# axs[1, 0].legend()
# axs[1, 0].grid(True)

# Plot 4: CE degradation vs k (Dict size = 12288)
# for sae_type in ['batchtopk', 'topk', 'jumprelu']:
#     data = df[(df['config_sae_type'] == sae_type) & (df['dictionary_size'] == 12288)]
#     axs[1, 1].plot(data['l0_norm'], data['ce_degradation'], 
#                    marker='o', linestyle='--', label=sae_type)

# axs[1, 1].set_title('CE degradation vs k (Dict size = 12288)')
# axs[1, 1].set_xlabel('k')
# axs[1, 1].set_ylabel('CE degradation')
# axs[1, 1].legend()
# axs[1, 1].grid(True)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
# %%
