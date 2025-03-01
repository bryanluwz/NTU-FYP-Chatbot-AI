from argparse import ArgumentParser
from flask import Flask
from flask_cors import CORS

from chat_routes import chat_bp
from chat_controller import init_chat_config
from dotenv import dotenv_values

config = dotenv_values(".env")
port_number = config.get('PORT', 5000)


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
    parser.add_argument('--api-mode', action='store_true',
                        help='Run the application in API mode')
    parser.add_argument('--port', type=int, default=port_number,
                        help='Port number to run the server on')
    args = parser.parse_args()

    init_chat_config({"debug": args.debug or False, "api_mode":
                      args.api_mode or False})

    app.run(host='0.0.0.0', port=args.port, ssl_context=(
        'server.cert', 'server.key'))
