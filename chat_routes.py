from flask import Blueprint

from chat_controller import query, stt, transferDocumentSrc, tts

chat_bp = Blueprint('chat', __name__)

chat_bp.add_url_rule('/query', view_func=query, methods=['POST'])
chat_bp.add_url_rule('/transferDocumentSrc',
                     view_func=transferDocumentSrc, methods=['POST'])
chat_bp.add_url_rule('/tts', view_func=tts, methods=['POST'])

chat_bp.add_url_rule('/stt', view_func=stt, methods=['POST'])
