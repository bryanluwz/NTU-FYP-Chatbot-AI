from flask import Blueprint

from chat_controller import query, transferDocumentSrc

chat_bp = Blueprint('chat', __name__)

chat_bp.add_url_rule('/query', view_func=query, methods=['POST'])
chat_bp.add_url_rule('/transferDocumentSrc',
                     view_func=transferDocumentSrc, methods=['POST'])
