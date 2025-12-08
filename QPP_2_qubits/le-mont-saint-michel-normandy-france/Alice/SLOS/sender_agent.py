#----------------------------------------------------------------------------------------------
# Alain Chancé April 29 2025
# 
# Alice client instantiates a SenderAgent and sends a file with the send_file method using:
#📡 requests module
#📦 File content encoded in Base64
#🌐 JSON-RPC 2.0 over HTTP
# 
# A2A leverages JSON-RPC 2.0 as the data exchange format for communication
# between a Client and a Remote Agent.
# https://google.github.io/A2A/#/documentation?id=agent-to-agent-communication
#-----------------------------------------------------------------------------------------------

import requests
import json
import base64
import sys

server_url="http://127.0.0.1:8000/rpc"

class SenderAgent:

    #---------------------------
    # Define ping_server method
    #---------------------------
    def ping_server(self, server_url=server_url) -> bool:
        
        # Build JSON-RPC 2.0 request
        payload = {
            "jsonrpc": "2.0",
            "method": "ping",
            "params": {},
            "id": 1
        }

        # Send the request
        try:
            response = requests.post(server_url, json=payload)
        except requests.exceptions.RequestException as e:
            print("\nSenderAgent - ping - Server not reachable:", e)
            return False

        # Show the response
        print("Status:", response.status_code)
        print("Server response:", response.json())
        return True

    #-------------------------
    # Define send_file method
    #-------------------------
    def send_file(self, filename="sample.txt", server_url=server_url) -> dict:

        # Read and encode the file content
        with open(filename, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode()

        # Build JSON-RPC 2.0 request
        payload = {
            "jsonrpc": "2.0",
            "method": "receive_file",
            "params": {
                "filename": filename,
                "content_b64": content_b64
            },
            "id": 1
        }

        # Send the request
        try:
            response = requests.post(server_url, json=payload)
        except:
            print("\nSenderAgent - send_file - Ensure that Bob_agent.ipynb has been run\n")
            print(sys.exc_info())
            return

        # Show the response
        print("Status:", response.status_code)
        print("Server response:", response.json())

        return