from flask import Flask, request, render_template
import torch
from sae_lens import SAE, HookedSAETransformer, ActivationsStore
from visualize import load_model_and_sae, get_dashboard_data

app = Flask(__name__)

# Globals to hold loaded model and SAE objects.
loaded_model = None
loaded_sae = None
loaded_device = None

# A dictionary for SAE options passed to the load_form template.
sae_options = {
    "tiny-stories-1L-21M": ["sae_ex32", "sae", "4", "5", "6", "7", "8", "16"],
    "gpt2-small": [f"blocks.{i}.hook_mlp_out" for i in range(12)]
}

@app.route("/", methods=["GET", "POST"])
def dashboard():
    global loaded_model, loaded_sae, loaded_act_store

    if request.method == "POST":
        action = request.form.get("action")
        # Step 1: Load model & SAE.
        if action == "load":
            model_name = request.form.get("model")
            sae_id = request.form.get("sae")
            loaded_model, loaded_sae, loaded_act_store = load_model_and_sae(model_name, sae_id)
            message = f"Model '{model_name}' and SAE '{sae_id}' loaded successfully."
            return render_template("dashboard.html", message=message, model_name=model_name,
                                   sae_id=sae_id, output=None)
        # Step 2: Run dashboard using loaded model & SAE.
        elif action == "run":
            prompt = request.form.get("prompt")
            model_name = request.form.get("model")
            sae_id = request.form.get("sae")
            # Ensure the model and SAE are loaded.
            if not loaded_model or not loaded_sae:
                loaded_model, loaded_sae, loaded_act_store = load_model_and_sae(model_name, sae_id)
            print(prompt)
            output = get_dashboard_data(loaded_model, loaded_sae, loaded_act_store, prompt)
            message = f"Model '{model_name}' and SAE '{sae_id}' are loaded. Dashboard results below:"
            return render_template("dashboard.html", message=message, model_name=model_name,
                                   sae_id=sae_id, output=output, prompt = prompt)
    # Default view: initial load form.
    return render_template("load_form.html", sae_options=sae_options)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)