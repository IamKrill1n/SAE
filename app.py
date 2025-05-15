from flask import Flask, render_template, request, jsonify
from visualize import load_stuffs, display_dashboard
import io
from contextlib import redirect_stdout
import re

app = Flask(__name__)

# Global variables to store loaded models and components
loaded_model = None
loaded_sae = None
loaded_act_store = None
loaded_proj = None
model_loaded = False

@app.route('/', methods=['GET', 'POST'])
def index():
    global loaded_model, loaded_sae, loaded_act_store, loaded_proj, model_loaded
    
    # Define SAE ID options based on model
    gpt2_sae_ids = [f"blocks.{i}.hook_mlp_out" for i in range(12)]
    tiny_stories_sae_ids = ["sae_ex32", "sae", "4", "5", "6", "7", "8", "16"]
    
    return render_template('index.html',
                         gpt2_sae_ids=gpt2_sae_ids,
                         tiny_stories_sae_ids=tiny_stories_sae_ids,
                         model_loaded=model_loaded)

@app.route('/load_model', methods=['POST'])
def load_model():
    global loaded_model, loaded_sae, loaded_act_store, loaded_proj, model_loaded
    
    model_name = request.form.get('model_name')
    sae_id = request.form.get('sae_id')
    
    try:
        loaded_model, loaded_sae, loaded_act_store, loaded_proj = load_stuffs(model_name, sae_id)
        model_loaded = True
        return jsonify({"status": "success", "message": f"Model {model_name} loaded successfully with SAE ID {sae_id}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    global loaded_model, loaded_sae, loaded_act_store, loaded_proj, model_loaded
    
    if not model_loaded:
        return jsonify({"status": "error", "message": "Please load the model first"})
    
    prompt = request.form.get('prompt')
    
    # Capture the output of display_dashboard
    output = io.StringIO()
    with redirect_stdout(output):
        display_dashboard(loaded_model, loaded_sae, loaded_act_store, loaded_proj, prompt)
    
    dashboard_output = output.getvalue()
    
    # Convert ANSI color codes to HTML
    dashboard_output = convert_ansi_to_html(dashboard_output)
    
    return jsonify({"status": "success", "output": dashboard_output})

def convert_ansi_to_html(text):
    # Convert [b u green]text[/] to HTML span with green color and bold
    text = re.sub(r'\[b u green\](.*?)\[/\]', r'<span style="color: green; font-weight: bold;">\1</span>', text)
    return text

if __name__ == '__main__':
    app.run(debug=True)