from flask import Blueprint

from chat_controller import query, query_voices, stt, transferDocumentSrc, tts, post_query_image

chat_bp = Blueprint('chat', __name__)

chat_bp.add_url_rule('/query', view_func=query, methods=['POST'])
chat_bp.add_url_rule('/transferDocumentSrc',
                     view_func=transferDocumentSrc, methods=['POST'])
chat_bp.add_url_rule('/tts', view_func=tts, methods=['POST'])

chat_bp.add_url_rule('/stt', view_func=stt, methods=['POST'])
chat_bp.add_url_rule(
    'postQueryImage', view_func=post_query_image, methods=['POST'])
chat_bp.add_url_rule('/availableVoices',
                     view_func=query_voices, methods=['GET'])
