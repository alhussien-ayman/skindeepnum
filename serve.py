from waitress import serve
from app import app  # Make sure this is your Flask app

serve(app, host="127.0.0.1", port=5000)