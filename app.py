from flask import Flask, render_template, request, redirect, url_for, flash
from visualize import load_stuffs, get_dashboard_data, display_dashboard

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

model = None
sae = None
activation_store = None
projection_onto_unembed = None

@app.route('/', methods=['GET', 'POST'])
def load_model():
    global model, sae, activation_store, projection_onto_unembed
    if request.method == 'POST':
        model_name = request.form['model_name']
        sae_id = request.form['sae_id']
        try:
            model, sae, activation_store, projection_onto_unembed = load_stuffs(model_name, sae_id)
            flash('Model and SAE loaded successfully!', 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash(f'Error loading model: {str(e)}', 'danger')
    return render_template('load.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        prompt = request.form['prompt']
        if model and sae and activation_store and projection_onto_unembed:
            dashboard_data = get_dashboard_data(model, sae, activation_store, projection_onto_unembed, prompt)
            return render_template('dashboard.html', dashboard_data=dashboard_data, prompt=prompt)
        else:
            flash('Model not loaded. Please load the model first.', 'danger')
            return redirect(url_for('load_model'))
    return render_template('dashboard.html', dashboard_data=None)

if __name__ == '__main__':
    app.run(debug=True)