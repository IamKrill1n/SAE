Capstone Project For NLP Course

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/IamKrill1n/SAE.git
    cd SAE
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


The `visualize.py` script requires two arguments:
- `--model_name`: The name of the model to visualize.
- `--sae_id`: The ID of the SAE to use.

Example usage:
    ```bash
    python visualize.py --model_name gpt2-small --sae_id blocks.7.hook_mlp_out
    ```