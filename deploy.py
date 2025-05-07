from flask import Flask, request, render_template_string
import torch
from sae_lens import SAE, HookedSAETransformer, ActivationsStore
from visualize import load_model_and_sae, get_dashboard_data

app = Flask(__name__)

# globals to hold loaded model and SAE objects
loaded_model = None
loaded_sae = None
loaded_device = None

# Template for the initial load form
load_form_template = """
<!doctype html>
<title>SAE Dashboard - Load Model & SAE</title>
<h1>Load Model & SAE</h1>
<form method="post">
  <input type="hidden" name="action" value="load">
  <label for="model">Model:</label>
  <select name="model" id="model">
    <option value="tiny-stories-1L-21M" selected>tiny-stories-1L-21M</option>
    <option value="gpt2-small">gpt2-small</option>
  </select>
  <br><br>
  <label for="sae">SAE ID:</label>
  <select name="sae" id="sae">
    {% if request.form.get("model") == "tiny-stories-1L-21M" %}
        <option value="sae_ex32" selected>sae_ex32</option>
        <option value="sae">sae</option>
        <option value="4">4</option>
        <option value="5">5</option>
        <option value="6">6</option>
        <option value="7">7</option>
        <option value="8">8</option>
        <option value="16">16</option>
    {% elif request.form.get("model") == "gpt2-small" %}
        {% for layer in range(12) %}
            <option value="blocks.{{ layer }}.hook_mlp_out">blocks.{{ layer }}.hook_mlp_out</option>
        {% endfor %}
    {% endif %}
  </select>
  <br><br>
  <input type="submit" value="Load Model & SAE">
</form>
"""

# Template for dashboard page after loading the model and SAE
dashboard_template = """
<!doctype html>
<title>SAE Dashboard</title>
<h1>{{ message }}</h1>
<hr>
<h2>Run Dashboard</h2>
<form method="post">
  <!-- Hidden fields preserve the loaded model selections -->
  <input type="hidden" name="action" value="run">
  <input type="hidden" name="model" value="{{ model_name }}">
  <input type="hidden" name="sae" value="{{ sae_id }}">
  
  <label for="prompt">Prompt:</label>
  <br>
  <textarea name="prompt" id="prompt" rows="3" cols="60">Once upon a time, there was a beautiful princess</textarea>
  <br><br>
  <input type="submit" value="Run Dashboard">
</form>
{% if output %}
<hr>
<h2>Dashboard Results</h2>
<p><b>Top Latents:</b> {{ output.top_latents }}</p>
<h3>Max Activating Examples for Neuron {{ output.top_latents[0] }}</h3>
<ul>
  {% for act, tokens, buff in output.max_examples %}
    <li><b>Activation:</b> {{ act }} | <b>Tokens:</b> {{ tokens }} | <b>Buffer:</b> {{ buff }}</li>
  {% endfor %}
</ul>
<h3>Top Logits for Top Latents:</h3>
<ol>
  {% for token in output.top_logits %}
    <li>{{ token }}</li>
  {% endfor %}
</ol>
{% endif %}
<p><a href="/">Reload Page</a></p>
"""

@app.route("/", methods=["GET", "POST"])
def dashboard():
    global loaded_model, loaded_sae, loaded_device

    if request.method == "POST":
        action = request.form.get("action")
        # Step 1: Load model & SAE
        if action == "load":
            model_name = request.form.get("model")
            sae_id = request.form.get("sae")
            loaded_model, loaded_sae, loaded_device = load_model_and_sae(model_name, sae_id)
            message = f"Model '{model_name}' and SAE '{sae_id}' loaded successfully on {loaded_device}."
            # Render dashboard template with prompt form; output is empty initially.
            return render_template_string(dashboard_template, message=message, model_name=model_name, sae_id=sae_id, output=None)

        # Step 2: Run dashboard using loaded model & SAE
        elif action == "run":
            prompt = request.form.get("prompt")
            model_name = request.form.get("model")
            sae_id = request.form.get("sae")
            # Ensure loaded_model and loaded_sae are loaded
            if not loaded_model or not loaded_sae:
                loaded_model, loaded_sae, loaded_act_store, loaded_device = load_model_and_sae(model_name, sae_id)
            # Get dashboard results for the prompt.
            output = get_dashboard_data(loaded_model, loaded_sae, loaded_act_store, loaded_device, prompt)
            message = f"Model '{model_name}' and SAE '{sae_id}' are loaded. Dashboard results below:"
            return render_template_string(dashboard_template, message=message, model_name=model_name, sae_id=sae_id, output=output)
    
    # Default view: initial load form
    return render_template_string(load_form_template)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)