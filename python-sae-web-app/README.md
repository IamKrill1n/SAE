# python-sae-web-app documentation

## Overview
This project is a web application that allows users to interact with a machine learning model and visualize its behavior based on user-defined prompts. Users can select a model and SAE ID, input prompts, and visualize the results in an intuitive dashboard.

## Project Structure
```
python-sae-web-app
├── app/
│   ├── __init__.py          # Initializes the Flask application
│   ├── routes.py            # Defines the application routes
│   ├── services.py          # Contains business logic for model loading and prompt processing
│   ├── utils.py             # Utility functions for tokenization and data formatting
│   ├── autointerp.py        # Imports functions for generating explanations
│   ├── static/              # Contains static files (CSS and JS)
│   │   ├── css/
│   │   │   └── style.css    # CSS styles for the web application
│   │   └── js/
│   │       └── script.js    # JavaScript for handling user interactions
│   └── templates/
│       └── index.html       # Main HTML template for the web application
├── run.py                   # Entry point for running the web application
├── requirements.txt         # Lists project dependencies
└── README.md                # Documentation for the project
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd python-sae-web-app
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Start the web application:
   ```
   python run.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000`.

3. Use the interface to:
   - Select a model name and SAE ID.
   - Enter a prompt in the provided text box.
   - Click on tokens to visualize their behavior and see the corresponding explanations.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.