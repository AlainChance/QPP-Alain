#-------------------------------------------------------------------------------
# Alain Chancé 29 April 2025
#
# Receiver agent functions as a uvicorn server and receives a file using:
#🛰️ FastAPI on the server (Remote Agent)
#📡 requests module on the client
#📦 File content encoded in Base64
#🌐 JSON-RPC 2.0 over HTTP
#
# uvicorn, https://www.uvicorn.org/
#
# FastAPI is a modern, fast (high-performance), web framework for building APIs 
# with Python based on standard Python type hints.
# https://fastapi.tiangolo.com/
#
# A2A leverages JSON-RPC 2.0 as the data exchange format for communication
# between a Client and a Remote Agent.
# https://google.github.io/A2A/#/documentation?id=agent-to-agent-communication
#
# JSON-RPC 2.0 Specification, https://www.jsonrpc.org/specification
#
# Welcome to jsonrpcserver’s documentation!
# https://www.jsonrpcserver.com/en/stable/examples.html#fastapi
#-------------------------------------------------------------------------------

from fastapi import FastAPI, Request
from jsonrpcserver import Result, Success, method, dispatch
import base64
import os
import uvicorn

app = FastAPI()

# os — Miscellaneous operating system interfaces
# https://docs.python.org/3/library/os.html
UPLOAD_DIR = "../Bob"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@method
def ping() -> Result:
    return Success(f"Receiver agent ready.")

@method
def receive_file(filename: str, content_b64: str) -> Result:
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(content_b64))
    return Success(f"File '{filename}' received and saved successfully.")

@app.post("/rpc")
async def rpc_handler(request: Request):
    response = dispatch(await request.body())
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)