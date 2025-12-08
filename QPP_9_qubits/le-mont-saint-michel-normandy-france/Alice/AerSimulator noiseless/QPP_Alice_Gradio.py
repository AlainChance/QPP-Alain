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

#-------------------------------------------------------------------------------------------------------------------------------------
# Install gradio
# Gradio is an open-source Python package that allows you to quickly build a demo or web application for your machine learning model, 
# API, or any arbitrary Python function. You can then share a link to your demo or web application in just a few seconds using 
# Gradio's built-in sharing features. No JavaScript, CSS, or web hosting experience needed! https://www.gradio.app/guides/quickstart
#-------------------------------------------------------------------------------------------------------------------------------------
import gradio as gr

import os
import shutil
import json

from QPP_Alain import QPP
from write_json import write_json
from sender_agent import SenderAgent

from qiskit_ibm_runtime import QiskitRuntimeService

#----------------------------------------------------------------------
# Define a function that returns: 
# - "None" and a list of real operational backends
# - "AerSimulator noiseless" and a list of corresponding fake backends
#----------------------------------------------------------------------
def list_backends(n_qubits=22):
    try:
        service = QiskitRuntimeService()
    except:
        service = None

    backend_options = ['AerSimulator noiseless', 'None', 'SLOS']
    
    if service is not None:
        try:
            backends = service.backends(None, min_num_qubits=n_qubits, simulator=False, operational=True)
        except Exception as e:
            backends = []
        
        backend_options += [f"fake_{backend.name[4:]}" for backend in backends]
        backend_options += [f"{backend.name}" for backend in backends]

    return backend_options

#------------------------------------------------------
# Define a function that process inputs from Gradio UI
#------------------------------------------------------
def process_inputs(
    # Parameters to write in Json file
    num_of_qubits,
    plaintext_file,
    version,
    trace,
    run_on_QPU,
    max_num_of_perm,
    backend_name,
    poll_interval,
    timeout, 
    # Other parameters
    run_simulation,
    save_alice_dir
):
    #------------------------------
    # Check type of plaintext file
    #------------------------------
    if not plaintext_file.lower().endswith((".txt", ".png", ".jpg")):
        return "Invalid file type. Please upload a .txt, .png, or .jpg file."

    #--------------------------------------------
    # Prepare parameters to write into Json file
    #--------------------------------------------
    backend_name = backend_name if run_on_QPU else "None"

    if backend_name != "AerSimulator noiseless" and not backend_name[:4] in ['None', 'SLOS', 'ibm_', 'fake']:
        return f"Invalid backend name {backend_name}"

    do_sampler = "True" if run_on_QPU else "False"

    max_num_of_perm = max_num_of_perm if backend_name[:4] in ['None', 'ibm_'] else 0
    
    version = "V" + str(version)

    #-------------------
    # Write JSON config
    #-------------------
    write_json(
        num_of_qubits=num_of_qubits,
        plaintext_file=plaintext_file,
        version=version,
        trace=trace,
        do_sampler=do_sampler,
        max_num_of_perm=max_num_of_perm,
        backend_name=backend_name,
        poll_interval=poll_interval,
        timeout=timeout
    )

    if run_simulation:
        #--------------------
        # Create Alice agent
        #--------------------
        server_url = "http://127.0.0.1:8000/rpc"
        alice_agent = SenderAgent()

        #-----------------
        # Ping server url
        #-----------------
        OK = alice_agent.ping_server(server_url=server_url)
        if not OK:
            return "Start a receiver agent with Bob_agent.ipynb"

        #--------------------
        # Quantum encryption
        #--------------------
        try:
            Alice_QPP = QPP("QPP_param")
        except Exception as e:
            return f"Error creating QPP instance: {e}"

        try:
            message = Alice_QPP.file_to_bitstring()
        except Exception as e:
            return f"Error sending message: {e}"

        try:
            ciphertext = Alice_QPP.encrypt(message=message)
        except Exception as e:
            return f"Error encrypting message: {e}"

        try:
            Alice_QPP.ciphertext_to_binary(ciphertext=ciphertext)
        except Exception as e:
            return f"Error converting cyphertext to binary: {e}"

        #-----------------------
        # Simulate transmission
        #-----------------------
        Secret_key_file = f"Secret_key_{num_of_qubits}-qubits_{plaintext_file[:-3]}txt"
        QPP_param_file = "QPP_param.json"
        ciphertext_file = f"ciphertext_{plaintext_file[:-3]}bin"
        trace_file = f"Trace_{num_of_qubits}-qubits_{plaintext_file[:-3]}txt"

        try:
            for fname in [Secret_key_file, QPP_param_file, ciphertext_file]:
                alice_agent.send_file(filename=fname, server_url=server_url)
        except Exception as e:
            return f"Error simulating transmission of secret key file, Json parameter file and cyphertext file: {e}"

    #-----------------------------------
    # Save Alice directory if requested
    #-----------------------------------
    if save_alice_dir:
        try:
            Alice_dir = f"../QPP_{num_of_qubits}_qubits/{plaintext_file[:-4]}/Alice/{backend_name}"
            os.makedirs(Alice_dir, exist_ok=True)
            for fname in [plaintext_file, Secret_key_file, ciphertext_file, trace_file,
                          "QPP_Alain.py", "sender_agent.py", "QPP_Alice_Gradio.py", "QPP_param.json",
                          "QPP_Alice.ipynb", "QPP_Alice.py", "Write_json.ipynb", "write_json.py"]:
                if os.path.isfile(fname):
                    shutil.copy(fname, Alice_dir)
            return f"Alice directory saved to: {Alice_dir}"
        except Exception as e:
            return f"Error saving Alice directory: {e}"

    if run_simulation:
        return "Encryption complete. Files transmitted."
    else:
        return "QPP configuration complete."

#-----------
# Gradio UI
#-----------
backend_options = list_backends()

backend_name = gr.Dropdown(
    label="Select noiseless, a real backend (or 'None' for least busy) or a fake backend",
    choices=backend_options,
    value="AerSimulator noiseless",
    visible=True  # visible by default
)
    
with gr.Blocks() as demo:
    gr.Markdown("## Configure And Run QPP Simulation")

    with gr.Row():
        num_of_qubits = gr.Slider(2, 9, step=1, label="Number of Qubits")
        max_num_of_perm = gr.Slider(0, 10, step=1, label="Max Permutations (only for specific backends)", value=0)
        poll_interval = gr.Slider(5, 20, step=5, label="Poll interval for job monitoring")
        timeout = gr.Slider(600, 1000, step=100, label="Time out for job monitoring")

    with gr.Row():
        plaintext_file = gr.Textbox(label="Plaintext file name (.txt, .png, .jpg)", value="le-mont-saint-michel-normandy-france.jpg")
        
        run_on_QPU = gr.Checkbox(label="Run on fake or real QPU ", value=True)
        backend_name.render()  # Render it outside the row, but keep it hidden initially
        run_on_QPU.change(
            fn=lambda checked: gr.update(visible=checked),
            inputs=run_on_QPU,
            outputs=backend_name
        )

    with gr.Row():
        version = gr.Radio([0, 1], label="Version (0 = n qubits, 1 = 2ⁿ qubits with swap gates)", value=1)
        trace = gr.Radio([0, 1], label="Trace Level", value=1)

    with gr.Row():
        run_simulation = gr.Checkbox(label="Run simulation", value=True)
        save_alice_dir = gr.Checkbox(label="Save Alice Directory", value=True)

    submit_btn = gr.Button("Configure QPP parameters and simulate secure transmission.")
    output = gr.Textbox(label="Status")

    submit_btn.click(
        fn=process_inputs,
        inputs=[
            # Parameters to write in Json file
            num_of_qubits,
            plaintext_file,
            version,
            trace,
            run_on_QPU,
            max_num_of_perm,
            backend_name,
            poll_interval,
            timeout,
            # Other parameters
            run_simulation,
            save_alice_dir
        ],
        outputs=output
    )

#-------------------------
# Launch Gradio interface
#-------------------------
demo.launch()