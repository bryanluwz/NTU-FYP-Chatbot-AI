from flask import Flask
from flask_cors import CORS
from chat_routes import chat_bp
from waitress import serve

# App stuff
app = Flask(__name__)
CORS(app)

app.register_blueprint(chat_bp, url_prefix='/api/chat')


@ app.route('/', methods=['GET'])
def home():
    return "Python Flask Server is running.", 200


if __name__ == '__main__':
    # Run the Flask app on port 5000
    # serve(app, host='0.0.0.0', port=5000, url_scheme='https')

    app.run(host='0.0.0.0', port=5000, ssl_context=(
        'server.cert', 'server.key'))
