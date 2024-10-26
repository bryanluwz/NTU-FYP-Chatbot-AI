from flask import Flask
from chat_routes import chat_bp

import torch


# App stuff
app = Flask(__name__)

app.register_blueprint(chat_bp, url_prefix='/api/chat')


@app.route('/', methods=['GET'])
def home():
    return "Python Flask Server is running.", 200


if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(host='0.0.0.0', port=5000)
