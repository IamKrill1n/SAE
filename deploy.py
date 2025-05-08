from flask import Flask, request, render_template
from sae_lens import SAE, HookedSAETransformer, ActivationsStore
from visualize import load_model_and_sae, get_dashboard_data

app = Flask(__name__)

# Globals to hold loaded model and SAE objects along with their names.
loaded_model = None
loaded_sae = None
loaded_act_store = None
loaded_model_name = None
loaded_sae_id = None

# A dictionary for SAE options passed to the load_form template.
sae_options = {
    "tiny-stories-1L-21M": ["sae_ex32", "sae", "4", "5", "6", "7", "8", "16"],
    "gpt2-small": [f"blocks.{i}.hook_mlp_out" for i in range(12)]
}

@app.route("/", methods=["GET", "POST"])
def dashboard():
    global loaded_model, loaded_sae, loaded_act_store, loaded_model_name, loaded_sae_id

    if request.method == "POST":
        action = request.form.get("action")
        # Step 1: Load model & SAE.
        if action == "load":
            model_name = request.form.get("model")
            sae_id = request.form.get("sae")
            loaded_model, loaded_sae, loaded_act_store = load_model_and_sae(model_name, sae_id)
            loaded_model_name = model_name
            loaded_sae_id = sae_id
            message = f"Model '{model_name}' and SAE '{sae_id}' loaded successfully."
            return render_template("dashboard.html", message=message, model_name=model_name,
                                   sae_id=sae_id, output=None, prompt="")
        # Step 2: Run dashboard using loaded model & SAE.
        elif action == "run":
            prompt = request.form.get("prompt")
            model_name = request.form.get("model")
            sae_id = request.form.get("sae")
            # If selected model or sae differ from currently loaded, reload them.
            if (loaded_model is None or loaded_sae is None or
                model_name != loaded_model_name or sae_id != loaded_sae_id):
                loaded_model, loaded_sae, loaded_act_store = load_model_and_sae(model_name, sae_id)
                loaded_model_name = model_name
                loaded_sae_id = sae_id
            output = get_dashboard_data(loaded_model, loaded_sae, loaded_act_store, prompt)
            message = f"Model '{model_name}' and SAE '{sae_id}' are loaded. Dashboard results below:"
            return render_template("dashboard.html", message=message, model_name=model_name,
                                   sae_id=sae_id, output=output, prompt=prompt)
    # Default view: initial load form.
    return render_template("load_form.html", sae_options=sae_options)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)