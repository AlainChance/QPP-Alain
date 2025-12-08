#--------------------------------------------------------------------------------
# Quantum Permutation Pad with Qiskit Runtime by Alain Chancé

## MIT License

# Copyright (c) 2022 Alain Chancé

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#--------------------------------------------------------------------------------

# This version features communications using JSON-RPC 2.0 over HTTP.
#
# The new Jupyter notebook Bob_agent.ipynb must be run first to start a receiver agent
# which functions as a uvicorn server and receives a file using:
# * 🛰️ FastAPI on the server (Remote Agent)
# * 📡 requests module on the client
# * 📦 File content encoded in Base64
# * 🌐 JSON-RPC 2.0 over HTTP
#
# Alice Jupyter notebook instantiates a SenderAgent and sends three files to Bob receiver agent with the send_file method using:
# * 📡 requests module
# * 📦 File content encoded in Base64
# * 🌐 JSON-RPC 2.0 over HTTP
#
# Last Bob Jupyter notebook process the files received from Alice sender agent.

## Documentation
#
# uvicorn, https://www.uvicorn.org/
# 
# FastAPI is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints.
# https://fastapi.tiangolo.com/
#
# A2A leverages JSON-RPC 2.0 as the data exchange format for communication between a Client and a Remote Agent.
# https://google.github.io/A2A/#/documentation?id=agent-to-agent-communication
#
# JS# ON-RPC 2.0 Specification, https://www.jsonrpc.org/specification
#
# Welcome to jsonrpcserver’s documentation!
# https://www.jsonrpcserver.com/en/stable/examples.html#fastapi
#

# os — Miscellaneous operating system interfaces
# https://docs.python.org/3/library/os.html
import os

# shutil — High-level file operations
# https://docs.python.org/3/library/shutil.html
import shutil

import json

# Import QPP_Alain
from QPP_Alain import QPP

QPP_param_file = "QPP_param.json"

if os.path.isfile(QPP_param_file):
    with open(QPP_param_file) as json_file:
        json_param = json.load(json_file)
else:
    raise RuntimeError("Missing QPP_param.json file")

num_of_qubits = json_param['num_of_qubits']

# Prompt for version 0 or 1
while True:
    try:
        v = int(input("Enter version 0 (n qubits) or 1 (2**n qubits which only uses swap gates): "))
        print("You entered: ", v)
        if v == 0 or v == 1:
            break
    except ValueError:
        print("Please enter a valid integer 0 or 1")

# Create an instance of the QPP class
Bob_QPP = QPP(QPP_param_file="QPP_param", version="V"+str(v))

# Read ciphertext binary file and extract the content to be transformed into a binary string
ciphertext = Bob_QPP.binary_to_ciphertext()

# Decrypt the ciphertext
decrypted_message = Bob_QPP.decrypt(ciphertext=ciphertext)

# Convert the decrypted message and save it into the decrypted file
Bob_QPP.bitstring_to_file(decrypted_message=decrypted_message)

print("\n\033[1mDo a File Save of this notebook\033[0m\n")

# Prompt the user
while True:
    response = input("Do you want to save the content of the Bob directory? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        print("You chose Yes.")
        plaintext_file = json_param['plaintext_file']
        Secret_key_file = "Secret_key_" + str(num_of_qubits) + "-qubits_" + plaintext_file[:-3] + "txt"
        ciphertext_file = "ciphertext_" + plaintext_file[:-3] + "bin"
        decrypted_file = "Decrypted_" + plaintext_file
        trace_file = "Trace_" + str(num_of_qubits) + "-qubits_" + plaintext_file[:-3] + "txt"

        backend_name = json_param['backend_name']
        Bob_dir = f"../QPP_{num_of_qubits}_qubits/{plaintext_file[:-4]}/Bob/{backend_name}"
        
        os.makedirs(Bob_dir, exist_ok=True)
        shutil.copy(Secret_key_file, Bob_dir)
        shutil.copy(ciphertext_file, Bob_dir)
        shutil.copy(decrypted_file, Bob_dir)
        shutil.copy(trace_file, Bob_dir)
        shutil.copy("QPP_Alain.py", Bob_dir)
        shutil.copy("receiver_agent.py", Bob_dir)
        shutil.copy("QPP_Bob.py", Bob_dir)
        shutil.copy("QPP_param.json", Bob_dir)
        shutil.copy("QPP_Bob.ipynb", Bob_dir)
        shutil.copy("Bob_agent.ipynb", Bob_dir)
        break
    elif response in ['n', 'no']:
        print("You chose No.")
        break
    else:
        print("Please enter 'y' or 'n'.")