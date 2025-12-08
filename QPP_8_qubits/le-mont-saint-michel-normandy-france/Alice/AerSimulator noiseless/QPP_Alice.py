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
import json
import subprocess

# shutil — High-level file operations
# https://docs.python.org/3/library/shutil.html
import shutil

# Import QPP_Alain
from QPP_Alain import QPP

from write_json import write_json

#------------------------------------------------------------------------
# Define the function QPP_configure that prompts the user for parameters
#------------------------------------------------------------------------
def QPP_configure():
    #-----------------------
    # Prompt for a filename
    #-----------------------
    while True:
        plaintext_file = input("Enter plaintext filename (.txt or .png or .jpg): ").strip()
        if plaintext_file.lower().endswith((".txt", ".png", ".jpg")):
            print("You entered: ", plaintext_file)
            if os.path.isfile(plaintext_file):
                break
            else:
                print("Missing plaintext file: ", plaintext_file)    
        else:
            print("Invalid file type. Please enter a .txt or .png or .jpg file.")

    #---------------------------------------------
    # Prompt for number of qubits between 2 and 9
    #---------------------------------------------
    while True:
        try:
            num_of_qubits = int(input("Enter number of qubits between 2 and 9: "))
            print("You entered: ", num_of_qubits)
            if num_of_qubits >= 2 and num_of_qubits <= 9:
                break
        except ValueError:
            print("Please enter a valid integer between 2 and 9")

    #---------------------------
    # Prompt for version 0 or 1
    #---------------------------
    while True:
        try:
            v = int(input("Enter version 0 (n qubits) or 1 (2**n qubits which only uses swap gates): "))
            print("You entered: ", v)
            if v == 0 or v == 1:
                break
        except ValueError:
            print("Please enter a valid integer 0 or 1")

    #-------------------------------
    # Prompt for trace level 0 or 1
    #-------------------------------
    while True:
        try:
            trace = int(input("Enter trace level 0 or 1: "))
            print("You entered: ", trace)
            if trace == 0 or trace == 1:
                break
        except ValueError:
            print("Please enter a valid integer 0 or 1")

    #---------------------------------------------------------------------
    # Prompt for Simulation Mode (0 = Classical, 1 = QPU or AerSimulator)
    #---------------------------------------------------------------------
    while True:
        try:
            user_int = int(input("Enter 0 for classical simulation or 1 for running on a real QPU or a simulation with AerSimulator"))
            print("You entered: ", user_int)
            if user_int == 0:
                do_sampler = "False"
                backend_name = "None"
            elif user_int == 1:
                do_sampler = "True"
            if user_int == 0 or user_int == 1:
                break
        except ValueError:
            print("Please enter a valid integer 0 or 1")

    if do_sampler == "True":
        #-----------------------------------------------------------------
        # Prompt for Backend Name (or 'None' or 'AerSimulator noiseless')
        #-----------------------------------------------------------------
        while True:
            backend_name = input("Enter IBM cloud backend name or 'None' to assign least busy device to backend or 'AerSimulator noiseless': ").strip()
            print("You entered: ", backend_name)
            if isinstance(backend_name, str):
                break
            else:
                print("Invalid backend_name. Please enter a character string.")

    max_num_of_perm = 0
    if do_sampler == "True" and backend_name[:4] in ['None', 'ibm_']:
        #--------------------------------------------------------------------------------
        # Prompt for maximum number of permutations in pad between 0 (no maximum) and 56
        #--------------------------------------------------------------------------------
        while True:
            try:
                max_num_of_perm = int(input("Enter maximum number of permutations in pad between 0 (no maximum) and 56: "))
                print("You entered: ", max_num_of_perm)
                if max_num_of_perm >= 0 and max_num_of_perm <= 56:
                    break
            except ValueError:
                print("Please enter a valid integer between 0 and 56")

    #-----------------
    # Write json file
    #-----------------
    write_json(num_of_qubits=num_of_qubits, 
               plaintext_file=plaintext_file, 
               version="V"+str(v), trace=trace, 
               do_sampler=do_sampler,
               max_num_of_perm=max_num_of_perm,
               backend_name=backend_name
              )

    return

#-------------------------------------------------------------
# Define the function QPP_simulation that runs the simulation
#-------------------------------------------------------------
def QPP_simulation():
    #-----------------------------------------
    # Retrieve parameters from QPP_param.json
    #-----------------------------------------
    QPP_param_file = "QPP_param.json"
    with open(QPP_param_file) as json_file:
        json_param = json.load(json_file)

    #--------------------
    # Quantum encryption
    #--------------------
    Alice_QPP = QPP("QPP_param")
    message = Alice_QPP.file_to_bitstring()
    ciphertext = Alice_QPP.encrypt(message=message)
    Alice_QPP.ciphertext_to_binary(ciphertext=ciphertext)

    #-----------------------
    # Simulate transmission
    #-----------------------
    plaintext_file = json_param['plaintext_file']
    num_of_qubits = json_param['num_of_qubits']

    Secret_key_file = f"Secret_key_{num_of_qubits}-qubits_{plaintext_file[:-3]}txt"
    ciphertext_file = f"ciphertext_{plaintext_file[:-3]}bin"
    trace_file = f"Trace_{num_of_qubits}-qubits_{plaintext_file[:-3]}txt"

    for fname in [Secret_key_file, QPP_param_file, ciphertext_file]:
        alice_agent.send_file(filename=fname, server_url=server_url)

    #--------------------------------------------------------------------
    # Prompt the user whether to save the content of the Alice directory
    #--------------------------------------------------------------------
    print("\n\033[1mDo a File Save of this notebook\033[0m\n")

    while True:
        response = input("Do you want to save the content of the Alice directory? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            print("You chose Yes.")
            backend_name = json_param['backend_name']
            Alice_dir = f"../QPP_{num_of_qubits}_qubits/{plaintext_file[:-4]}/Alice/{backend_name}"
            os.makedirs(Alice_dir, exist_ok=True)
            for fname in [plaintext_file, Secret_key_file, ciphertext_file, trace_file,
                          "QPP_Alain.py", "sender_agent.py", "QPP_Alice_Gradio.py", "QPP_param.json",
                          "QPP_Alice.ipynb", "QPP_Alice.py", "Write_json.ipynb", "write_json.py"]:
                if os.path.isfile(fname):
                     shutil.copy(fname, Alice_dir)
            print(f"Alice directory saved to: {Alice_dir}")
            break

        elif response in ['n', 'no']:
            print("You chose No.")
            break
        
        else:
            print("Please enter 'y' or 'n'.")

    return

#--------------------
# Create Alice agent
#--------------------
from sender_agent import SenderAgent
server_url="http://127.0.0.1:8000/rpc"
alice_agent = SenderAgent()
response_dict = {}

#-----------------
# Ping server url
#-----------------
OK = alice_agent.ping_server(server_url=server_url)
if not OK:
    while True:
        response = input("Start a receiver agent with Bob_agent.ipynb - and enter 'y' when done").strip()
        if response.lower() == "y":
            print("You entered: ", response)
            if alice_agent.ping_server(server_url=server_url):
                print("Receiver agent reached")
                break
            else:
                print("Receiver agent not reachable - try again")
        else:
            print("Enter y when done")

#-------------------------------------------
# Prompt whether to run Gradio UI interface
#-------------------------------------------
do_Gradio = True

while True:
    try:
        response = input("\nStart Gradio UI interface? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            print("You chose Yes.")
            do_Gradio = True
            break
        
        elif response in ['n', 'no']:
            print("You chose No.")
            do_Gradio = False
            break
    
    except ValueError:
        print("Invalid response. Please enter 'y' or 'n'.")


if not do_Gradio:
    #--------------------------------
    # QPP configuration with prompts
    #--------------------------------
    QPP_configure()
    
    #----------------
    # QPP simulation
    #----------------
    QPP_simulation()
        
else:
    #-----------------------------------------------
    # Launch the QPP_Alice_Gradio.py asynchronously
    #-----------------------------------------------
    print("Running Gradio UI on local URL: http://127.0.0.1:7860")

    # Launch the script asynchronously
    process = subprocess.Popen(["python", "QPP_Alice_Gradio.py"],
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT)
    
    #----------------------------------------------------------
    # Prompt whether QPP configuration with Gradio is complete
    #----------------------------------------------------------
    while True:
        response = input("\nQPP configuration with Gradio complete? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            print("You chose Yes.")
            break
        elif response in ['n', 'no']:
            print("You chose No.")
        else:
            print("Invalid response. Please enter 'y' or 'n'.")
    #----------------------------------
    # Prompt whether to run simulation 
    #----------------------------------
    while True:
        response = input("\nRun QPP simulation? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            print("You chose Yes.")
            QPP_simulation()
            break
        elif response in ['n', 'no']:
            print("You chose No.")
            break
        else:
            print("Invalid response. Please enter 'y' or 'n'.")
