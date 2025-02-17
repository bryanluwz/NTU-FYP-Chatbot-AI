from argparse import ArgumentParser
from flask import Flask
from flask_cors import CORS

from chat_routes import chat_bp
from chat_controller import init_chat_config


# App stuff
app = Flask(__name__)
CORS(app)

app.register_blueprint(chat_bp, url_prefix='/api/chat')


@app.route('/', methods=['GET'])
def home():
    return "Python Flask Server is running.", 200


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                        help='Run the application in debug mode')
    args = parser.parse_args()

    init_chat_config({"debug": args.debug or False})

    app.run(host='0.0.0.0', port=5000, ssl_context=(
        'server.cert', 'server.key'))
