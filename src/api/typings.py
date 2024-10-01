from pydantic import BaseModel


class RequestModel(BaseModel):
    user_input: str  # User input (query)
    model_name: str  # To specify different LLM models


class ResponseModel(BaseModel):
    response: str  # Respond with a string (can include stringified data)
