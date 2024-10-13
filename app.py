from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/query', methods=['POST'])
def process_request():
    data = request.get_json()
    print(data)
    response = {
        'data': {
            'response': 'Received the data successfully!'
        }
    }
    return jsonify(response), 200


@app.route('/', methods=['GET'])
def home():
    return "Python Flask Server is running.", 200


if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(host='0.0.0.0', port=5000)
