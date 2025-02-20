# NTU FYP Chatbot AI

This repository contains the AI server implementation for the NTU Final Year Project (FYP) Chatbot. This AI server is built using Python and Flask for processing chatbot logic. This AI server powers the chatbot's core functionalities, providing a seamless and intelligent user experience.

## Warning

I might have messed up some of the code trying to get the Docker setup to work. If you encounter any issues, please let me know and I will not fix it, because I'm done with this project.

## Features

Key features of the AI server include:

1. **Chat Query Reformulation**: Enhances user queries before passing them to an LLM.

2. **Retrieval-Augmented Generation (RAG)**: Dynamically determines when to retrieve additional information based on query similarity.

3. **Speech-to-Text & Text-to-Speech**: Uses Hugging Face models for STT/TTS capabilities.

4. **Streaming Support**: Handles long-running AI tasks asynchronously.

5. **Logging and Monitoring**: Logs AI activities and errors for monitoring and debugging purposes.

## Setup and Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/bryanluwz/NTU-FYP-Chatbot-AI.git
   ```

2. Install the dependencies:

   This project uses Python 3.10.14.

   Make sure to have some virtual environment set up. If you don't, you can create one using:

   ```bash
   python -m venv venv
   ```

   Then, activate the virtual environment:

   ```bash
   # On Windows
   .\venv\Scripts\Activate.ps1
   # On Unix or MacOS
   source ./venv/bin/activate
   ```

   You can also use `conda` if you prefer. But I won't provide instructions for that. 😛

   ```bash
   pip install -r requirements.txt
   ```

   Some of the dependencies are useless because I might have forgotten to remove them. You can try remove one-by-one and see if the server still works. If you actually do this, please let me know the results.

3. Create your own `server.key` and `server.cert` files for HTTPS:

   ```bash
   openssl req -nodes -new -x509 -keyout server.key -out server.cert
   ```

   This command, which might or might not work on Windows, will generate a self-signed SSL certificate and private key. You can also use a valid SSL certificate if you have one, or already obtained from the [backend repository](https://github.com/bryanluwz/NTU-FYP-Chatbot-backend).

   Don't know how? Too bad, figure it out yourself!

4. Run the AI server:

   ```bash
   python ./app.py
   # or with debug logging
   python ./app.py --debug
   ```

5. The AI server should now be running on `https://localhost:3001` or whatever port you specified in the `.env` file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

1. [ME](https://github.com/bryanluwz) for building this awesome chatbot frontend, alone, with no help from humans. 🤖
