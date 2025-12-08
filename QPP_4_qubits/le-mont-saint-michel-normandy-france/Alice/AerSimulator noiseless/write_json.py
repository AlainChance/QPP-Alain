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

# This version has been updated to work with the upgraded IBM Quantum Platform, https://quantum.cloud.ibm.com/

# Table 1 - The table tabulates the equivalent Shannon information entropy per n-qubit permutation space for n from 2 to 5.
# Source: https://epjquantumtechnology.springeropen.com/articles/10.1140/epjqt/s40507-022-00145-y/tables/1
# From: Quantum encryption with quantum permutation pad in IBMQ systems. 
# https://epjquantumtechnology.springeropen.com/articles/10.1140/epjqt/s40507-022-00145-y

# os — Miscellaneous operating system interfaces
# https://docs.python.org/3/library/os.html
import os

import json

def write_json(
    QPP_param_file = "QPP_param.json",
    num_of_qubits = 2,
    plaintext_file = "Christmas_tree.png",
    version = "V0",
    trace = 1,
    do_sampler = "True",
    max_num_of_perm = 0,
    backend_name = None,
    poll_interval = 5,
    timeout = 600
):

    file_size = os.path.getsize(plaintext_file)
    print(f"Plaintext file size: {file_size} bytes")

    job_trigger = int(file_size/10) + 1
    print_trigger = job_trigger

    trace = trace

    if num_of_qubits < 6:
        match num_of_qubits:
            case 2:
                num_of_bits = 448
                num_of_perm_in_pad = 56
            case 3:
                num_of_bits = 408
                num_of_perm_in_pad = 17
            case 4:
                num_of_bits = 384
                num_of_perm_in_pad = 6
            case 5:
                num_of_bits = 480
                num_of_perm_in_pad = 3
    else:
        num_of_perm_in_pad = 3
        n = 2**num_of_qubits
        num_of_bits = (n-1) * num_of_qubits * num_of_perm_in_pad * 2

    if max_num_of_perm > 0:
        num_of_perm_in_pad = min(num_of_perm_in_pad, max_num_of_perm)
        
    json_param = {
        "num_of_bits": num_of_bits,
        "num_of_qubits": num_of_qubits,
        "num_of_perm_in_pad": num_of_perm_in_pad,
        "pad_selection_key_size": 6,
        "opt_level": 1,
        "resilience_level": 1,
        "plaintext_file": plaintext_file,
        "token_file": "Token.txt",
        "CRN_file": "CRN.txt",
        "trace": trace,
        "job_trigger": job_trigger,
        "print_trigger": print_trigger,
        "draw_circuit": "True",
        "do_sampler": do_sampler,
        "backend_name": backend_name,
        "version": version,
        "len_message": 12758,
        "len_ciphertext": 102064,
        "poll_interval": poll_interval,
        "timeout": timeout
    }
    
    # Serializing json
    json_object = json.dumps(json_param, indent=4)
 
    # Write to QPP_param.json
    with open(QPP_param_file, 'w') as json_file:
        json_file.write(json_object)

    return