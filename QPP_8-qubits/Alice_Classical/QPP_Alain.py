# Quantum Permutation Pad with Qiskit Runtime by Alain Chancé

## MIT License

# MIT_License Copyright (c) 2022 Alain Chancé
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the \"Software\"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.'
# THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.'

## Abstract

# We demonstrate an efficient implementation of the Kuang and Barbeau’s Quantum Permutation pad (QPP) 
# symmetric cryptographic algorithm with Qiskit Runtime, a new architecture offered by IBM Quantum 
# that streamlines quantum computations. We have implemented a Python class QPP and template Jupyter 
# notebooks with Qiskit code for encrypting and decrypting with n-qubit QPP any text file in UTF-16 format
# or any image file in .png format. We offer the option of running either a quantum circuit with n qubits,
# or an alternate one with 2n qubits which only uses swap gates and has a circuit depth of O(n). 
# It is inherently extremely fast and could be run efficiently on currently available noisy quantum computers.
# Our implementation leverages the new Qiskit Sampler primitive in localized mode which dramatically 
# improves performance. We offer a highly efficient classical implementation which performs permutation gate 
# matrix multiplication with information state vectors. We illustrate the use with two agents Alice and Bob
# who exchange a text file and an image file using 2-qubit QPP and 4-qubit QPP. 

# Keywords—quantum communication, quantum encryption,
# quantum decryption, quantum security, secure communication, QPP, Qiskit, IBMQ

## Credit: Kuang, R., Perepechaenko

# Appendix, source code for the implementation of the 2-qubits QPP of the following article:
# Kuang, R., Perepechaenko, M. Quantum encryption with quantum permutation pad in IBMQ systems. 
# EPJ Quantum Technol. 9, 26 (2022). https://doi.org/10.1140/epjqt/s40507-022-00145-y

# This article is licensed under a Creative Commons Attribution 4.0 International License.

## Adaptations made by Alain Chancé
#
# Summary of updates V3
# This jupyter notebook has been updated to work with Qiskit 1.3.1 
#
# Updates in Install statements
# Previously
# from qiskit.primitives import Sampler
# New
# from qiskit.primitives import StatevectorSampler
#
## Updates in code
# In class QPP
#
# Commented out
# #options = Options(resilience_level=resilience_level)
# Previously
# sampler = Sampler()
# New
# sampler = StatevectorSampler()
# 
# Updated the with Sesssion according to the example given in the Session documentation:
# https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.Session
# with Session(backend=backend) as session:
#
#--------------------------------------------------------------------------------------------------------------------------------
# Summary of updates V2
# This jupyter notebook has been updated to work with Qiskit 1.0.2 

# Please refer to the following documentation:
# - Qiskit 1.0 feature changes, https://docs.quantum.ibm.com/api/migration-guides/qiskit-1.0-features#qiskit-10-feature-changes
# - Qiskit 1.0 migration tool, https://docs.quantum.ibm.com/api/migration-guides/qiskit-1.0-features#qiskit-10-migration-tool
# - Qiskit Aer 0.14.0.1, Getting started, https://qiskit.github.io/qiskit-aer/getting_started.html
# - Migrate setup from qiskit-ibmq-provider, https://docs.quantum.ibm.com/api/migration-guides/qiskit-runtime#migrate-setup-from-qiskit-ibmq-provider
# - Simulators, The AerSimulator, https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html

## Updates in install statements
### Install qiskit.aer
# New
# https://qiskit.github.io/qiskit-aer/getting_started.html
# pip install qiskit_aer

## Updates in import statements
### Import AerSimulator
# New
# from qiskit_aer import AerSimulator

### Migrate setup from qiskit-ibmq-provider
# Previously
# from qiskit import IBMQ
# New
# from qiskit_ibm_runtime import QiskitRuntimeService

## Updates in code
### Updates in permutation_pad() function
# New
# aer_sim = AerSimulator(method='statevector')
#...
# qci = transpile(qci, aer_sim)
# job = aer_sim.run(qci, shots=1024)
# result = job.result()
# counts = result.get_counts(qci)

#--------------------------------------------------------------------------------------------------------------------------------

# Summary of updates V1

# Please refer to the following documentation:

# [3] Introducing new Qiskit Runtime capabilities — and how our clients are integrating them into # their use, https://research.ibm.com/blog/qiskit-runtime-capabilities-integration

# [4] Qiskit Runtime IBM Client, https://github.com/Qiskit/qiskit-ibm-runtime

# [5] Qiskit IBM runtime, Configure error mitigation, Advanced resilience options, 
# https://github.com/Qiskit/qiskit-ibm-
# runtime/blob/ab7486d6837652d54cb60b83cfaa9165f5d0484c/docs/how_to/error-
# mitigation.rst#advanced-resilience-options

### New class QPP
#
# class QPP:
#    def __init__(self, QPP_param_file = "QPP_param.json"):
#--------------------------------------------------------------------------------------------------------------------------------

# Import NumPy
import numpy as np

#------------------------------------------------------------------------------------------------------------------------
# Import qiskit.aer
# Additional circuit methods. On import, Aer adds several simulation-specific methods to QuantumCircuit for convenience. 
# These methods are not available until Aer is imported (import qiskit_aer). 
# https://qiskit.github.io/qiskit-aer/apidocs/circuit.html
#------------------------------------------------------------------------------------------------------------------------
import qiskit_aer

#---------------------
# Import AerSimulator
#---------------------
from qiskit_aer import AerSimulator

#-----------------------------
# Import QiskitRuntimeService
#-----------------------------
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options

#---------------------------------------------
# Import the localized version of the Sampler
#---------------------------------------------
#from qiskit.primitives import Sampler
from qiskit.primitives import StatevectorSampler

#-------------------------------------------------
# Import the required functions and class methods
#-------------------------------------------------
# Importing standard Qiskit libraries
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile

from qiskit.visualization import plot_histogram, plot_bloch_vector, array_to_latex

from qiskit.circuit.library import RXGate, XGate, CXGate, SwapGate

import qiskit.quantum_info as qi
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import process_fidelity
from qiskit.quantum_info import Statevector

#--------------------------------------
# Import any other necessary libraries
#--------------------------------------
from random import seed, randint
import time, datetime
import psutil
import sys
import os

#-------------------------
# Import the JSON package
#-------------------------
import json

#---------------------------------------------
### Import Bitstream, BitArray
# https://bitstring.readthedocs.io/en/stable/
#---------------------------------------------
from bitstring import BitStream, BitArray

#------------------------------------------------------------------------------------------------
# If the import "from QPP_Alain import QPP" fails and file QPP_Alain.py is in the same directory 
# as your python or Jupyter notebook, try adding the following lines:
# import sys
# import os
# cwd = os.getcwd()
# _= (sys.path.append(cwd))
#------------------------------------------------------------------------------------------------

class QPP:
    def __init__(self, QPP_param_file = "QPP_param"):
        # Initialize json_param, param to None
        self.json_param = None
        self.param = None
    
        # Open QPP_param JSON file and save content in json_param dictionary
        QPP_param_file = QPP_param_file + ".json"
        if os.path.isfile(QPP_param_file):
            with open(QPP_param_file) as json_file:
                self.json_param = json.load(json_file)
        else:
            print("parameters - Error: missing QPP_param.json file: {}".format(QPP_param_file))
            return
        
        json_param = self.json_param
        
        # Get parameters from json_param dictionary
        num_of_bits = json_param['num_of_bits'] # Classical key length (bit)
        num_of_qubits = json_param['num_of_qubits'] # Number of qubits
        num_of_perm_in_pad = json_param['num_of_perm_in_pad'] # Number of permutation matrices in pad
        pad_selection_key_size = json_param['pad_selection_key_size'] # Pad selection key size
        opt_level = json_param['opt_level']
        resilience_level = json_param['resilience_level'] # Resilience level, check Ref [4]
        plaintext_file = json_param['plaintext_file']
        trace = json_param['trace'] # Trace level 0, 1, or 2
        token_file = json_param['token_file'] # token for QiskitRuntimeService
        job_trigger = json_param['job_trigger'] # Number of iterations after which a job sampler.run(circuits=list_qc) is submitted
        print_trigger = json_param['print_trigger'] # Number of iterations after which a print is done
        draw_circuit = (json_param['draw_circuit'] == 'True') # Set to 'True' to draw transpiled circuits
        do_sampler = (json_param['do_sampler'] == 'True') # Set to 'True' to use Sampler() if enough memory is available
        version = json_param['version'] # Version of the quantum circuit used to simulate a permutation operation
        len_message = json_param['len_message'] # True length of ciphertext set by file_to_bitstring()
        len_ciphertext = json_param['len_ciphertext'] # True length of ciphertext set by encrypt()
        
        Intro_0 = "\033[1m"+"Quantum permutation Pad with Qiskit Runtime by Alain Chancé"+"\033[0m"
        
        MIT_License = "\n"+"\033[1m"+"MIT License"+"\033[0m"+\
        '\nCopyright (c) 2022 Alain Chancé'+\
'\nPermission is hereby granted, free of charge, to any person obtaining a copy\n\
of this software and associated documentation files (the \"Software\"), to deal\n\
in the Software without restriction, including without limitation the rights\n\
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n\
copies of the Software, and to permit persons to whom the Software is\n\
furnished to do so, subject to the following conditions:\n\n\
The above copyright notice and this permission notice shall be included in all\n\
copies or substantial portions of the Software.'+\
'\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n\
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n\
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n\
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n\
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n\
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n\
SOFTWARE.'
        
        Intro_1 = "\n"+"\033[1m"+"Abstract"+"\033[0m"+\
        "\nWe demonstrate an efficient implementation of the Kuang and Barbeau’s Quantum Permutation pad (QPP) symmetric cryptographic \n\
algorithm with Qiskit Runtime, a new architecture offered by IBM Quantum that streamlines quantum computations. We have \n\
implemented a Python class QPP and template Jupyter notebooks with Qiskit code for encrypting and decrypting with n-qubit QPP \n\
any text file in UTF-16 format or any image file in .png format. We offer the option of running either a quantum circuit \n\
with n qubits, or an alternate one with 2**n qubits which only uses swap gates and has a circuit depth of O(n). \n\
It is inherently extremely fast and could be run efficiently on currently available noisy quantum computers. Our implementation \n\
leverages the new Qiskit Sampler primitive in localized mode which dramatically improves performance. We offer a highly \n\
efficient classical implementation which performs permutation gate matrix multiplication with information state vectors. We \n\
illustrate the use with two agents Alice and Bob who exchange a text file and an image file using 2-qubit QPP and 4-qubit QPP.\n\
\nKeywords—quantum communication, quantum encryption, quantum decryption, quantum security, secure communication, QPP, Qiskit, \n\
IBMQ"
        Intro_2 = "\n"+"\033[1m"+"Credit"+"\033[0m"+\
        "\nKuang, R., Perepechaenko, Appendix, source code for the implementation of the 2-qubits QPP of the following article:\n\
Kuang, R., Perepechaenko, M. Quantum encryption with quantum permutation pad in IBMQ systems. EPJ Quantum Technol. 9, 26 (2022). \
https://doi.org/10.1140/epjqt/s40507-022-00145-y"+\
"\n\n"+"\033[1m"+"Rights and Permissions"+"\033[0m"+"\n"+"\033[1m"+"Open Access"+"\033[0m"+\
" This article is licensed under a Creative Commons Attribution 4.0 International License.\n\
To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/."
        
        Intro_3 = "\n"+"\033[1m"+"References"+"\033[0m"+"\n"\
"[1] R. Kuang and N. Bettenburg, 'Shannon perfect secrecy in a discrete Hilbert space', in Proc. IEEE Int. Conf. Quantum Comput. Eng. (QCE),\
Oct. 2020, pp. 249-255, doi: 10.1109/QCE49297.2020.00039\n\
[2] Kuang, R., Perepechaenko, M. Quantum encryption with quantum permutation pad in IBMQ systems. EPJ Quantum Technol. 9, 26 (2022).\
 https://doi.org/10.1140/epjqt/s40507-022-00145-y\n\
[3] Introducing new Qiskit Runtime capabilities — and how our clients are integrating them into their use, \
https://research.ibm.com/blog/qiskit-runtime-capabilities-integration\n\
[4] Qiskit Runtime IBM Client, https://github.com/Qiskit/qiskit-ibm-runtime\n\
[5] Qiskit IBM runtime, Configure error mitigation, Advanced resilience options, \
https://github.com/Qiskit/qiskit-ibm-runtime/blob/ab7486d6837652d54cb60b83cfaa9165f5d0484c/docs/how_to/error-mitigation.rst#advanced-resilience-options\n\
[6] Fisher–Yates shuffle, Wikipedia, https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle\n\
[7] Qiskit Terra API Reference, Primitives, Sampler, https://qiskit.org/documentation/stubs/qiskit.primitives.Sampler.html\n\
[8] Qiskit Terra API Reference, Primitives, Source code for qiskit.primitives.sampler, \
https://qiskit.org/documentation/_modules/qiskit/primitives/sampler.html#Sampler\n\
[9] Qiskit, qiskit.quantum_info.Statevector.from_instruction, https://qiskit.org/documentation/stubs/qiskit.primitives.Sampler.html\n\
[10] Qiskit, Source code for qiskit.quantum_info.states.statevector, Statevector.from_instruction, \
https://qiskit.org/documentation/_modules/qiskit/quantum_info/states/statevector.html#Statevector.from_instruction"
        
        Intro_4 = "\n"+"\033[1m"+"Trademarks"+"\033[0m"+"\n"\
"IBM®, IBM Q Experience®, and Qiskit® are registered trademarks of IBM Corporation."

        Intro_5 = "\n" +"\033[1m"+"Using QPP_Alain.py"+"\033[0m"+"\n"\
        "To use QPP_Alain, put QPP_Alain.py in the same directory as your code and insert the following import:\n\
from QPP_Alain import QPP"
    
        # Print introduction text with abstract, credit and using QPP_Alain.py
        print(Intro_0)
        print(MIT_License)
        print(Intro_1)
        print(Intro_2)
        print(Intro_3)
        print(Intro_4)
        print(Intro_5)
        
        if trace > 0:
            print("\nParameters read from file: {}".format(QPP_param_file))
            print(self.json_param)
    
        # Create or open a trace file
        trace_file = "Trace_" + plaintext_file[:-3] + "txt"
        trace_f = open(trace_file,"w") # Open trace file in write mode
        if trace > 0:
            print("Printing trace into file: ", trace_file)
        
        # Set n to 2 to the power of the number of qubits
        n = 2**num_of_qubits
    
        if trace > 0:
            print("\nSet n, possible number of quantum states = 2**num_of_qubits: {}".format(n), file=trace_f)
        
        # This implementation uses the localized version of the `Sampler` which resides in Qiskit Terra [7][8]
        # and uses the statevector simulator to compute probabilities:
        #
        #   probabilities = [
        #         Statevector(bound_circuit_to_instruction(circ)).probabilities(qargs=qargs)
        #        for circ, qargs in zip(bound_circuits, qargs_list)]
        #
        # Statevector simulator requires 2**instruction.num_qubits data of type complex [9][10]:
        #
        #     Initialize an the statevector in the all |0> state
        #     init = np.zeros(2**instruction.num_qubits, dtype=complex)
        #
        # Versions "V0" and "V1" of the quantum circuit that perform a permutation operation use the following number of qubits: 
        # - V0: N = num_of_qubits
        # - V1: N = n = 2**num_of_qubits
    
        # Let's compute the amount of memory required to store 2**N numbers of data type complex128
        if version == "V0":
            N = num_of_qubits
        elif version == "V1":
            N = n
        else:
            print("parameters - unknown version: {} should be either V0 or V1".format(version))
            return
    
        # Amount of memory required to store one quantum circuit that simulates a permutation operation
        # https://numpy.org/doc/stable/reference/arrays.dtypes.html
        mem_circuit = np.dtype(np.complex128).itemsize*8*2**N
    
        # Amount of memory required to store all the circuits in the pad
        mem_req = num_of_perm_in_pad * mem_circuit
    
        # Get available memory for processes
        # https://www.geeksforgeeks.org/how-to-get-current-cpu-and-ram-usage-in-python/
        mem_avail = psutil.virtual_memory()[1]
    
        # Check that there is enough available memory for the statevector simulation
        if mem_req > mem_avail:
        
            print("\nparameters - Not enough memory is available to use Statevector simulator")
            print("Amount of memory required by Statevector simulator to simulate one circuit (GB) {}".format(mem_circuit/10**9))
            print("Number of permutation matrices in pad: {}".format(num_of_perm_in_pad))
            print("Amount of memory required (GB): {}, amount of available memory for processes (GB) {}"
                .format(mem_req/10**9, mem_avail/10**9))
            print("Using classical computation, multiplying matrix of permutation operator with state vector")
        
            print("\nNot enough memory is available to use Statevector simulator", 
                  file=trace_f)
            print("Amount of memory required by Statevector simulator to simulate one circuit (GB) {}".format(mem_circuit/10**9),
                 file=trace_f)
            print("Number of permutations in pad: {}".format(num_of_perm_in_pad), file=trace_f)
            print("Amount of memory required (GB): {}, amount of available memory for processes (GB) {}"
                .format(mem_req/10**9, mem_avail/10**9), file=trace_f)
            print("Using classical computation, multiplying matrix of permutation operator with state vector",
                 file=trace_f)
        
            # Set do_sampler to False
            do_sampler = False
        
        # Initialize service, backend and options to None
        service = None
        backend = None
        options = None
        
        if do_sampler:
            # Test if token file exists
            if os.path.isfile(token_file):
                f = open(token_file, "r") 
                token = f.read() # Read token from token_file
                print("Token read from file: ", token_file)
                f.close()
                
                # Save the Qiskit Runtime account credentials
                QiskitRuntimeService.save_account(channel='ibm_quantum', token=token, overwrite=True)

                # https://docs.quantum.ibm.com/api/migration-guides/qiskit-runtime
                service = QiskitRuntimeService(channel='ibm_quantum')
                
                # Define a local backend
                # https://docs.quantum.ibm.com/api/migration-guides/qiskit-runtime-from-ibm-provider
                # https://docs.quantum.ibm.com/api/migration-guides/qiskit-runtime-from-ibm-provider#get-a-backend
                from qiskit_ibm_runtime.fake_provider import FakeManilaV2
                backend = FakeManilaV2()
                if trace > 0:
                    print("\nparameters - Backend", file=trace_f)
                    print(backend, file=trace_f)
                
                # Check Ref. [5]
                #options = Options(resilience_level=resilience_level)
            
            else:
                print("parameters - Error: required token file with do_sampler = 'True' is missing {}".format(token_file))
                return
        
        # Set print_trigger to be the largest of job_trigger and print_trigger
        print_trigger = max(print_trigger, job_trigger)
    
        # Set basis gates
        basis_gates=["u3","u2","u1","cx","id","u0","u","p","x","y","z","h","s",
                 "sdg","t","tdg","rx","ry","rz","sx","sxdg","cz","cy","swap",
                "ch","ccx","cswap","crx","cry","crz","cu1","cp","cu3","csx",
                "cu","rxx","rzz"]
    
        # Get secret key file name
        secret_key_file = "Secret_key_" + str(num_of_qubits) + "-qubits_" + plaintext_file[:-3] + "txt"
        if trace > 0:
            print("\nparameters - Secret key file name: {}\n".format(secret_key_file), file=trace_f)
    
        # Test if secret key file exists
        if os.path.isfile(secret_key_file):
            f = open(secret_key_file,"r") # Open secret key file
            secret_key = f.read() # Retrieve secret key from file
            f.close()
            if trace > 0:
                print("parameters - Secret key retrieved from file: ", secret_key, file=trace_f)
        else:
            secret_key = self.rand_key(num_of_bits) # Create a truly random key
            f = open(secret_key_file,"w") # Open secret key file
            f.write(secret_key ) # Store secret key into the secret key file
            f.close()
            if trace > 0:
                print("parameters - Secret key created and stored into file: ", secret_key, file=trace_f)
    
        # Compute bits_in_block as number of bits / number of permutations gates in pad
        bits_in_block = int(num_of_bits / num_of_perm_in_pad)
        if trace > 0:
            print("\nparameters - Set bits_in_block = num_of_bits / num_of_perm_in_pad: {}".format(bits_in_block), file=trace_f)
    
        # Check the consistency of n, possible number of quantum states and the secret key length, bits_in_block/
        len_key_chunks = int(bits_in_block/num_of_qubits)
    
        if trace > 0:
            print("\nparameters - Length of key chunks (bits_in_block/num_of_qubits): {}".format(len_key_chunks), file=trace_f)
    
        if n-1 >= len_key_chunks:
            print("parameters - Error: n-1 (possible number of quantum states-1): {} >= length of key chunks (bits_in_block/num_of_qubits): {}"
                  .format(n-1, len_key_chunks), file=trace_f)
            return
    
        # Set param dictionary
        self.param = {'QPP_param_file':QPP_param_file, 'plaintext_file':plaintext_file, 'trace':trace, 'trace_file':trace_file, 
             'job_trigger': job_trigger, 'print_trigger': print_trigger, 'secret_key': secret_key, 'n':n,
             'num_of_bits':num_of_bits, 'bits_in_block':bits_in_block, 'num_of_qubits':num_of_qubits, 
             'num_of_perm_in_pad':num_of_perm_in_pad, 'pad_selection_key_size':pad_selection_key_size, 
             'service':service, 'backend':backend, 'options':options, 'opt_level':opt_level, 'basis_gates':basis_gates, 
             'len_message':len_message,'len_ciphertext':len_ciphertext, 'draw_circuit': draw_circuit, 'do_sampler': do_sampler,
             'version': version}
    
        if trace > 0:
            print("QPP_param_file: {}".format(QPP_param_file), file=trace_f)
            print("plaintext_file: {}".format(plaintext_file), file=trace_f)
            print("trace: {}".format(trace), file=trace_f)
            print("job_trigger: {}".format(job_trigger), file=trace_f)
            print("print_trigger: {}".format(print_trigger), file=trace_f)
            print("draw_circuit: {}".format(draw_circuit), file=trace_f)
            print("do_sampler: {}".format(do_sampler), file=trace_f)
            print("version: {}".format(version), file=trace_f)
            print("", file=trace_f)
            print("service: {}".format(service), file=trace_f)
            print("backend: {}".format(backend), file=trace_f)
            print("opt_level, optimisation level for transpile: {}".format(opt_level), file=trace_f)
            print("resilience_level: {}".format(resilience_level), file=trace_f)
            print("", file=trace_f)
            print(options, file=trace_f)
            print("", file=trace_f)
            print("n, possible number of quantum states: {}".format(n), file=trace_f)
            print("num_of_bits, classical key length (bit): {}".format(num_of_bits), file=trace_f)
            print("bits_in_block: {}".format(bits_in_block), file=trace_f) # number of bits / number of permutations gates in pad
            print("num_of_qubits: {}".format(num_of_qubits), file=trace_f)
            print("num_of_perm_in_pad, number of permutations gates in pad: {}".format(num_of_perm_in_pad), file=trace_f)
            print("pad_selection_key_size: {}".format(pad_selection_key_size), file=trace_f)
    
        trace_f.close()
    
        return
    
    ##############################################################################################################
    # Define the function rand_key() to create a random binary string                                            #
    #                                                                                                            # 
    # https://www.geeksforgeeks.org/python-program-to-generate-random-binary-string/                             #
    ##############################################################################################################
    def rand_key(self, p):
   
        # Variable to store the string
        key1 = ""
 
        # Loop to find the string of desired length
        for i in range(p):
         
            # randint function to generate 0, 1 randomly and converting the result into str
            temp = str(randint(0, 1))
 
            # Concatenation the random 0, 1 to the final result
            key1 += temp
    
        return(key1)
    
    ##############################################################################################################
    # Define randomize() function to perform Fisher Yates shuffling                                              #
    # The Fisher–Yates shuffle is an algorithm for generating a random permutation of a finite sequence [5].     #
    ##############################################################################################################
    def randomize(self, arr, n, key_chunks, qc, qr, num_of_perm, transpose=False):
        
        if self.param == None:
            print("randomize: missing parameter param")
            return arr
    
        param = self.param
    
        trace = param['trace']
        trace_file = param['trace_file']
        version = param['version']
        do_sampler = param['do_sampler']
    
        trace_f = open(trace_file,"a") # Open trace file in append mode
    
        # Sanity check added by Alain Chancé
        if n - 1 >= len(key_chunks):
            print("randomize - Error: n-1: {} >= len(key_chunks): {}".format(n-1, len(key_chunks)))
            arr = None
            trace_f.close()
            return arr
    
        if trace > 1:
            print("\nrandomize - Permutation number: {}".format(num_of_perm), file=trace_f)
    
        if transpose:
            for i in range(1, n):
                j = key_chunks[i]
        
                if trace > 1:
                    print("randomize - Permuting columns {} and {}".format(i, j), file=trace_f)
        
                arr[i], arr[j] = arr[j], arr[i]
        
                if version == "V1" and do_sampler:
                    # Add swap gate
                    if i != j:
                        qc.swap(qr[i], qr[j])
    
        else:
            for i in range(n-1,0,-1):
                j = key_chunks[i]
        
                if trace > 1:
                    print("randomize - Permuting columns {} and {}".format(i, j), file=trace_f)
        
                arr[i], arr[j] = arr[j], arr[i]
        
                if version == "V1" and do_sampler:
                    # Add swap gate
                    if i != j:
                        qc.swap(qr[i], qr[j])
    
        trace_f.close()
    
        return arr
    
    ##############################################################################################################
    # Define permutation_pad() function                                                                          #
    # The Fisher–Yates shuffle is an algorithm for generating a random permutation of a finite sequence [6].     #
    ##############################################################################################################
    def permutation_pad(self, secret_key_blocks, transpose=False):
        
        Permutation_Pad = None
        perm_dict = None
        
        if secret_key_blocks == None:
            print("permutation_pad: missing parameter secret_key_blocks")
            return Permutation_Pad, perm_dict
        
        if self.param == None:
            print("permutation_pad: missing param")
            return Permutation_Pad, perm_dict
        
        param = self.param
    
        trace = param['trace']
        trace_file = param['trace_file']
    
        trace_f = open(trace_file,"a") # Open trace file in append mode
    
        if trace > 0:
            print("\npermutation_pad", file=trace_f)
        
        Permutation_Pad = []
        n = param['n']
        num_of_perm_in_pad = param['num_of_perm_in_pad']
        num_of_qubits = param['num_of_qubits']
        service = param['service']
        backend = param['backend']
        options = param['options']
        opt_level = param['opt_level']
        basis_gates = param['basis_gates']
        job_trigger = param['job_trigger']
        print_trigger = param['print_trigger']
        draw_circuit = param['draw_circuit'] # Set to True to draw quantum circuits
        do_sampler = param['do_sampler'] # Set to True to use Sampler() if enough memory is available
        version = param['version']
    
        if version == "V0":
            N = num_of_qubits
        elif version == "V1":
            N = n
        else:
            print("\npermutation_pad - unknown version")
            return Permutation_Pad, perm_dict
        
        # This implementation uses the localized version of the `Sampler` which resides in Qiskit Terra [7][8]
        # and uses the statevector simulator to compute probabilities:
        #
        #   probabilities = [
        #         Statevector(bound_circuit_to_instruction(circ)).probabilities(qargs=qargs)
        #        for circ, qargs in zip(bound_circuits, qargs_list)]
        #
        # Statevector simulator requires 2**instruction.num_qubits data of type complex [9][10]:
        #
        #     Initialize an the statevector in the all |0> state
        #     init = np.zeros(2**instruction.num_qubits, dtype=complex)
        #
        # Versions "V0" and "V1" of the quantum circuit that perform a permutation operation use the following number of qubits: 
        # - V0: N = num_of_qubits
        # - V1: N = n = 2**num_of_qubits
    
        # Let's compute the amount of memory required to store 2**N numbers of data type complex128
        
        # Amount of memory required to store one quantum circuit that simulates a permutation operation
        # https://numpy.org/doc/stable/reference/arrays.dtypes.html
        mem_circuit = np.dtype(np.complex128).itemsize*8*2**N
    
        # Amount of memory required to store all the circuits in the pad
        mem_req = num_of_perm_in_pad * mem_circuit
    
        # Get available memory for processes
        # https://www.geeksforgeeks.org/how-to-get-current-cpu-and-ram-usage-in-python/
        mem_avail = psutil.virtual_memory()[1]
    
        # Check that there is enough available memory for the statevector simulation
        if mem_req > mem_avail:
        
            print("\npermutation_pad - Not enough memory is available to use Statevector simulator")
            print("Amount of memory required by Statevector simulator to simulate one circuit (GB) {}".format(mem_circuit/10**9))
            print("Number of permutations in pad: {}".format(num_of_perm_in_pad))
            print("Amount of memory required (GB): {}, amount of available memory for processes (GB) {}"
                .format(mem_req/10**9, mem_avail/10**9))
            print("Using classical computation, multiplying matrix of permutation operator with state vector")
        
            print("\nNot enough memory is available to use Statevector simulator", 
                  file=trace_f)
            print("Amount of memory required by Statevector simulator to simulate one circuit (GB) {}".format(mem_circuit/10**9),
                 file=trace_f)
            print("Number of permutations in pad: {}".format(num_of_perm_in_pad), file=trace_f)
            print("Amount of memory required (GB): {}, amount of available memory for processes (GB) {}"
                .format(mem_req/10**9, mem_avail/10**9), file=trace_f)
            print("Using classical computation, multiplying matrix of permutation operator with state vector",
                 file=trace_f)
        
            # Set do_sampler to False
            do_sampler = False
        
        M = np.eye(n)
        perm_dict = []
        
        start = time.time()

        aer_sim = AerSimulator(method='statevector')
        
        if do_sampler:
            # Create a session constructor using Qiskit Runtime Session()
                with Session(backend=backend) as session:
                    #sampler = Sampler()
                    sampler = StatevectorSampler()
        
                    for num_of_perm in range(num_of_perm_in_pad):
        
                        # Prepare a quantum circuit for the permutation
                        qr = QuantumRegister(N, 'q')
                        cr = ClassicalRegister(N, 'c')
                        qc = QuantumCircuit(qr, cr)
        
                        # Build an array of elements from 0 to n-1
                        my_array = []
                        for num in range(n):
                            my_array.append(num)
                        array_of_n_num = my_array.copy()
    
                        # Create a shuffled array using Fisher-Yates and secret key blocks
                        key_block = secret_key_blocks[num_of_perm]
        
                        key_chunks = [key_block[i:i+num_of_qubits] for i in range(0, len(key_block), num_of_qubits)]
    
                        for num in range(len(key_chunks)):
                            key_chunks[num] = int(key_chunks[num],2)
        
                        len_of_array = len(my_array)
                        shuffled_array = self.randomize(my_array, len_of_array, key_chunks, qc, qr, num_of_perm, transpose=transpose)

                        # create a matrix of zeros
                        matrix_of_zeros = np.zeros((n, n), dtype=int)
                        my_matrix = matrix_of_zeros
    
                        # insert '1' at every column but different rows
                        for num in range(n):
                            my_matrix[array_of_n_num[num]][shuffled_array[num]] = 1
    
                        # Populate the Permutation Pad with permutation matrices converted to quantum operators
                        Permutation_Pad.append(Operator(my_matrix))
                        
                        if trace > 0:
                            print("\npermutation pad - Permutation number: {}, matrix: ".format(num_of_perm))
                            display(array_to_latex(my_matrix))
                        
                        if version == "V0":
                        # Apply permutation
                            qc.append(Operator(my_matrix), range(N))
                            qc.barrier()
                            qc = transpile(qc, basis_gates=basis_gates, optimization_level=opt_level)
                            
                        if trace > 0:            
                            print("\npermutation pad - Permutation number: {}, Depth of quantum circuit: {}".
                                format(num_of_perm, qc.depth()), file=trace_f)
                            print("\npermutation pad - Permutation number: {}, Depth of quantum circuit: {}".
                                    format(num_of_perm, qc.depth()))
                            if draw_circuit:
                                display(qc.draw(output='mpl'))

                        # Create a list of dictionaries of most frequent
                        pdict = {}
                        
                        # Run the quantum circuit for every possible input state
                        for i in range(2**num_of_qubits):
                            
                            # Establish a circuit with N qubits and N classical bits to store measurement results later
                            qri = QuantumRegister(N, 'qi')
                            cri = ClassicalRegister(N, 'ci')
                            qci = QuantumCircuit(qri, cri)
            
                            # Initialize the qubits according to the randomized message bits
                            state_vector = eval("format(i, '0" + str(num_of_qubits) + "b')")
                            if version == "V0":
                                if trace > 1:
                                    print("Initializing version V0, i: {}, state_vector: {}".format(i, state_vector))
                                qci.initialize(Statevector.from_label(state_vector))
                            else:
                                if trace > 1: 
                                    print("Initializing version V1, i: {}, state_vector: {}".format(i, state_vector))
                                qci.x(qri[int(state_vector, 2)])
                            
                            # Append permutation circuit
                            qci.append(qc, range(N), range(N))
            
                            # Measure all qubits
                            qci.measure(qri, cri)
                            
                            if trace > 1:            
                                print("\npermutation pad - Permutation number: {}, State_vector: {}, Depth of quantum circuit: {}".
                                    format(num_of_perm, state_vector, qci.depth()), file=trace_f)
                                print("\npermutation pad - Permutation number: {}, State_vector: {}, Depth of quantum circuit: {}".
                                        format(num_of_perm, state_vector, qci.depth()))
                                display(qci.draw(output='mpl'))

                            qci = transpile(qci, aer_sim)
                            job = aer_sim.run(qci, shots=1024)
                            result = job.result()
                            counts = result.get_counts(qci)
                            
                            if version == "V0":
                                # Get dictionary key with the max value
                                maxkey = max(counts, key=counts.get)
                                most_frequent = maxkey
                                pdict[i] = most_frequent
                            
                                if trace > 1:
                                    print("permutation_pad - k: {}, most_frequent: {}".format(k, most_frequent))
                            
                            else:  # Version V1
                                # Get dictionary key with the max value
                                maxkey = max(counts, key=counts.get)
                                ix = n - maxkey.index('1') - 1
                                most_frequent = eval("format(ix, '0" + str(num_of_qubits) + "b')")
                                pdict[i] = most_frequent
                        
                                if trace > 1:
                                    print("permutation_pad - k: {}, maxkey: {}, ix: {}, most_frequent: {}".
                                        format(k, maxkey, ix, most_frequent))
                    
                        # Append pdict to the list perm_dict
                        perm_dict.append(pdict)  
                        
                        # Print permutation dictionary
                        if trace > 0:
                            elapsed = str(datetime.timedelta(seconds = time.time()-start))
                            
                            print("\npermutation_pad - permutation number: {}, dictionary:".format(num_of_perm), file=trace_f)
                            print(pdict, file=trace_f)
                            print("permutation pad - Elapsed time: {}".format(elapsed), file=trace_f)
                        
                            print("\npermutation_pad - permutation number: {}, dictionary:".format(num_of_perm))
                            print(pdict)
                            print("permutation pad - Elapsed time: {}".format(elapsed))
            
        else: #Classical processing
            for num_of_perm in range(num_of_perm_in_pad):
        
                # Build an array of elements from 0 to n-1
                my_array = []
                for num in range(n):
                    my_array.append(num)
                array_of_n_num = my_array.copy()
    
                # Create a shuffled array using Fisher-Yates and secret key blocks
                key_block = secret_key_blocks[num_of_perm]
        
                key_chunks = [key_block[i:i+num_of_qubits] for i in range(0, len(key_block), num_of_qubits)]
    
                for num in range(len(key_chunks)):
                    key_chunks[num] = int(key_chunks[num],2)
        
                len_of_array = len(my_array)
                shuffled_array = self.randomize(my_array, len_of_array, key_chunks, None, None, num_of_perm, transpose=transpose)

                # create a matrix of zeros
                matrix_of_zeros = np.zeros((n, n), dtype=int)
                my_matrix = matrix_of_zeros
    
                # insert '1' at every column but different rows
                for num in range(n):
                    my_matrix[array_of_n_num[num]][shuffled_array[num]] = 1
    
                # Populate the Permutation Pad with permutation matrices converted to quantum operators
                Permutation_Pad.append(Operator(my_matrix))
            
                # Create a dictionary of most frequent
                pdict = {}
                    
                # Compute classically the output state vector by multiplying permutation matrix with input state vector
                for k in range(n):
                    s = M[k]
                    ix = (my_matrix@s).tolist().index(1)
                    most_frequent = eval("format(ix, '0" + str(num_of_qubits) + "b')")
                    pdict[k] = most_frequent
            
                # Print permutation dictionary
                if trace > 0:
                    elapsed = str(datetime.timedelta(seconds = time.time()-start))
                    
                    print("\npermutation_pad - permutation number: {}, dictionary:".format(num_of_perm), file=trace_f)
                    print(pdict, file=trace_f)
                    print("permutation_pad - Elapsed time: {}".format(elapsed), file=trace_f)
                        
                    print("\npermutation_pad - permutation number: {}, dictionary:".format(num_of_perm))
                    print(pdict)
                    print("permutation_pad - Elapsed time: {}".format(elapsed))
                
                # Append pdict to the list perm_dict
                perm_dict.append(pdict)
        
        if trace > 0:
            print("permutation pad - Length of Permutation_Pad: {}\n".format(len(Permutation_Pad)), file=trace_f)
            print("permutation pad - Length of Permutation_Pad: {}\n".format(len(Permutation_Pad)))
    
        trace_f.close()
    
        return Permutation_Pad, perm_dict
    
    ##############################################################################################################
    # Define the function encrypt() to encrypt a message                                                         #
    ##############################################################################################################
    def encrypt(self, message=None):
        
        ciphertext = None
        
        if message == None:
            print("encrypt - missing parameter message")
            return ciphertext
        
        if self.param == None:
            print("encrypt: missing param")
            return ciphertext
            
        if self.json_param == None:
            print("encrypt: missing json_param")
            return ciphertext
        
        json_param = self.json_param
        param = self.param
        
        QPP_param_file = param['QPP_param_file']
        n = param['n']
        secret_key = param['secret_key']
        trace = param['trace']
        trace_file = param['trace_file']
        bits_in_block = param['bits_in_block']
        pad_selection_key_size = param['pad_selection_key_size']
        num_of_qubits = param['num_of_qubits']
        num_of_perm_in_pad = param['num_of_perm_in_pad']
        service = param['service']
        backend = param['backend']
        options = param['options']
        opt_level = param['opt_level']
        basis_gates = param['basis_gates']
        job_trigger = param['job_trigger']
        print_trigger = param['print_trigger']
        draw_circuit = param['draw_circuit'] # Set to True to draw quantum circuits
        do_sampler = param['do_sampler'] # Set to True to use Sampler() if enough memory is available
        version = param['version']
    
        trace_f = open(trace_file,"a") # Open trace file in append mode
    
        len_message = len(message)
    
        if trace > 0:
            print("encrypt - Length of message in bits: ", len_message, file=trace_f)
            print("encrypt - Length of message in bits: ", len_message)
        
           
        # Check that the length of the message is a multiple of the number of qubits
        r = len_message % num_of_qubits # Remainder of the length of message divided by the number of qubits
        if r != 0: # If length of message is not a multiple of the number of qubits
            print("encrypt - Error: Length of message {} is not a multiple of the number of qubits {}".
                  format(len_message, num_of_qubits), file=trace_f)
            ciphertext = None
            trace_f.close()
            return ciphertext
    
        # Break the secret key into blocks of size bits_in_block
        secret_key_blocks = [secret_key[i:i+bits_in_block] for i in range(0, len(secret_key), bits_in_block)]
    
        # Create a permutation pad
        Permutation_Pad, perm_dict = self.permutation_pad(secret_key_blocks, transpose=False)
    
        # Create an array used for dispatching
        pad_selection_blocks = [secret_key[i:i+pad_selection_key_size] for i in range(0, len(secret_key), pad_selection_key_size)]

        for num in range(len(pad_selection_blocks)):
            pad_selection_blocks[num] = (int(pad_selection_blocks[num],2)%num_of_perm_in_pad)

        # Define secret key blocks to be used for randomization
        key_for_xor_list = []

        for x in range(len(message)):
            key_for_xor_list.append(secret_key[x%len(secret_key)])
    
        key_for_xor = "".join(key_for_xor_list)
    
        # Randomize the message using XOR operation
        randomized_message = [str(int(message[i])^int(key_for_xor[i])) for i in range(len(message))]
        randomized_message = ''.join(randomized_message)
    
        # Break the randomized message into blocks of the number of qubits
        chunk_size = num_of_qubits
        message_chunks = [randomized_message[i:i+chunk_size] for i in range(0, len(randomized_message), chunk_size)]
    
        #------------------------------------------
        # Initialize variables
        start = time.time()
        count_qc = 0
        count = 0
        list_of_ciphers = []
        list_qc = []
        list_x = []
        list_j = []
        list_state_vector = []
        list_most_frequent = []
        len_message_chunks = len(message_chunks)
        r1 = len_message_chunks % job_trigger
        M = np.eye(n)
        #-------------------------------------------
    
        if trace > 0:
            print("encrypt - Length of randomized message: ", len(randomized_message), file=trace_f)
            print("encrypt - Length of message chunks: {}".format(len_message_chunks), file=trace_f)
            print("encrypt - Remainder of dividing (Length of message chunks) by (Job trigger): {}".format(r1), file=trace_f)
            print("", file=trace_f)
            
        # Classical processing  
        for x in range(len_message_chunks):
            state_vector = message_chunks[x]
            
            # Apply permutation from the pad based on its position in the pad determined by the dispatching array
            j = pad_selection_blocks[x%len(pad_selection_blocks)]
                
            # Get most_frequent from the dictionary
            pdict = perm_dict[j]
            most_frequent = pdict[int(state_vector, 2)]
                
            # ************ Note list of ciphers *****************
            list_of_ciphers.append(most_frequent)
                
            if trace > 1:
                list_most_frequent.append(most_frequent)
                print("encrypt - Permutation_Pad[{}], most_frequent: {}".format(j, most_frequent))        
            
            if (trace == 1 and count == print_trigger - 1):
                elapsed = str(datetime.timedelta(seconds = time.time()-start))
                
                print("encrypt - x : {},  Permutation_Pad[{}], State vector: {}, Most frequent: {}".
                        format(x, j, state_vector, most_frequent), file=trace_f)
                print("encrypt - Elapsed time: {}".format(elapsed), file=trace_f)
                
                print("encrypt - x : {},  Permutation_Pad[{}], State vector: {}, Most frequent: {}".
                        format(x, j, state_vector, most_frequent))
                print("encrypt - Elapsed time: {}".format(elapsed))
                
                count = 0
            else:
                count += 1
    
        if trace > 0:
            elapsed = str(datetime.timedelta(seconds = time.time()-start))
            
            print("\nencrypt - Elapsed time for encryption of message: {}".format(elapsed), file=trace_f)
            print("\nencrypt - Elapsed time for encryption of message: {}".format(elapsed))
    
        # Create one unified ciphertext string
        ciphertext = "".join(list_of_ciphers)
    
        # Save length of ciphertext in json_param dictionary, enabling parameters() to retrieve it and store it into param
        json_param['len_ciphertext'] = len(ciphertext)
    
        # Serializing json
        json_object = json.dumps(json_param, indent=4)
 
        # Write to QPP_param.json
        with open(QPP_param_file, 'w') as json_file:
            json_file.write(json_object)
    
        # Save length of ciphertext in param dictionary, enabling decrypt() to retrieve it without calling parameters() first
        param['len_ciphertext'] = len(ciphertext)

        if trace > 0:
            print("\nencrypt - Length of ciphertext in bits stored into QPP_param.json: ", len(ciphertext), file=trace_f)
        
            print("\nencrypt - First 192 bits in ciphertext string", file=trace_f)
            print(ciphertext[:min(len(ciphertext), 192)], file=trace_f)
        
            print("\nencrypt - First 192 bits in ciphertext string")
            print(ciphertext[:min(len(ciphertext), 192)])
    
        trace_f.close()
    
        return ciphertext
    
    ##############################################################################################################
    # Define the function decrypt() to decrypt a message                                                         #
    ##############################################################################################################
    def decrypt(self, ciphertext=None):
        
        decrypted_message = None
    
        if ciphertext == None:
            print("decrypt - missing parameter ciphertext")
            return decrypted_message
        
        if self.param == None:
            print("randomize: missing param")
            return decrypted_message
    
        param = self.param
    
        n = param['n']
        secret_key = param['secret_key']
        trace = param['trace']
        trace_file = param['trace_file']
        bits_in_block = param['bits_in_block']
        pad_selection_key_size = param['pad_selection_key_size']
        num_of_qubits = param['num_of_qubits']
        num_of_perm_in_pad = param['num_of_perm_in_pad']
        service = param['service']
        backend = param['backend']
        options = param['options']
        opt_level = param['opt_level']
        basis_gates = param['basis_gates']
        job_trigger = param['job_trigger']
        print_trigger = param['print_trigger']
        len_ciphertext = param['len_ciphertext'] # retrieve true length of ciphertext set by encrypt()
        draw_circuit = param['draw_circuit'] # Set to True to draw transpiled circuits
        do_sampler = param['do_sampler'] # Set to True to use Sampler() if enough memory is available
        version = param['version']
    
        trace_f = open(trace_file,"a") # Open trace file in append mode
    
        if trace > 0:
            # Note retrieve true length of ciphertext from param dictionary
            print("\ndecrypt - Length of cipher text in bits retrieved from param dictionary: {}".
                  format(len_ciphertext), file=trace_f)
            print("\ndecrypt - Length of cipher text in bits retrieved from param dictionary: {}".format(len_ciphertext))
        
            print("\ndecrypt - First 192 bits in ciphertext string", file=trace_f)
            print(ciphertext[:min(len_ciphertext, 192)], file=trace_f)
        
            print("\ndecrypt - First 192 bits in ciphertext string")
            print(ciphertext[:min(len_ciphertext, 192)])
        
        # Check that the length of ciphertext is a multiple of the number of qubits
        r = len_ciphertext % num_of_qubits # Remainder of the length of ciphertext divided by the number of qubits
        if r != 0: # If length of ciphertext is not a multiple of the number of qubits
            print("decrypt - Error: Length of ciphertext {} is not a multiple of the number of qubits {}".format(len_ciphertext, 
                                                            num_of_qubits), file=trace_f)
            trace_f.close()
            return decrypted_message
    
        # Break ciphertext into blocks
        chunk_size = num_of_qubits
    
        # We use true length of ciphertext set by encrypt(), not the length of the text read from the ciphertext binary file
        cipher_chunks = [ciphertext[i:i+chunk_size] for i in range(0, len_ciphertext, chunk_size)]
    
        # Break the secret key into blocks
        secret_key_blocks = [secret_key[i:i+bits_in_block] for i in range(0, len(secret_key), bits_in_block)]
    
        # Create an inverse permutation pad
        Inverse_Permutation_Pad, perm_dict = self.permutation_pad(secret_key_blocks, transpose=True)
    
        # Create an array used for dispatching
        pad_selection_blocks = [secret_key[i:i+pad_selection_key_size] for i in range(0, len(secret_key), pad_selection_key_size)]

        for num in range(len(pad_selection_blocks)):
            pad_selection_blocks[num] = (int(pad_selection_blocks[num],2)%num_of_perm_in_pad)
    
        # Define secret key blocks to be used for randomization
        key_for_xor_list = []

        for x in range(len_ciphertext):
            key_for_xor_list.append(secret_key[x%len(secret_key)])
    
        key_for_xor = "".join(key_for_xor_list)
        
        #--------------------------------------
        # Initialize variables
        start = time.time()
        count_qc = 0
        count = 0
        list_of_messages = []
        list_qc = []
        list_x = []
        list_j = []
        list_state_vector = []
        list_most_frequent = []
        len_cipher_chunks = len(cipher_chunks)
        r2 = len_cipher_chunks % job_trigger
        M = np.eye(n)
        #---------------------------------------
        
        if trace > 0:
            print("decrypt - Length of ciphertext: ", len_ciphertext, file=trace_f)
            print("decrypt - Remainder of dividing (Length of cipher chunks) by (Job trigger): {}".format(r2), file=trace_f)
            print("", file=trace_f)
            
            print("decrypt - Length of ciphertext: {}".format(len_ciphertext))
            print("decrypt - Remainder of dividing (Length of cipher chunks) by (Job trigger): {}".format(r2))
            print("")
        
        # Classical processing
        for x in range(len_cipher_chunks):
            state_vector = cipher_chunks[x]
            
            j = pad_selection_blocks[x%len(pad_selection_blocks)]
            
            # Get most_frequent from the dictionary
            pdict = perm_dict[j]
            most_frequent = pdict[int(state_vector, 2)]
                
            # ************ Note list of messages *****************
            list_of_messages.append(most_frequent)
            
            if trace > 1:
                print("decrypt - Permutation_Pad[{}], most_frequent: {}".format(j, most_frequent))
    
            if (trace == 1 and count == print_trigger - 1):
                elapsed = str(datetime.timedelta(seconds = time.time()-start))
                print("decrypt - x : {},  Permutation_Pad[{}], State vector: {}, Most frequent: {}".
                        format(x, j, state_vector, most_frequent), file=trace_f)
                print("decrypt - Elapsed time for decryption: {}".format(elapsed), file=trace_f)
                
                print("decrypt - x : {},  Permutation_Pad[{}], State vector: {}, Most frequent: {}".
                        format(x, j, state_vector, most_frequent))
                print("decrypt - Elapsed time for decryption: {}".format(elapsed))
                
                count = 0
            else:
                count += 1
            #------------------------------------------------------------------------------------
    
        randomized_decrypted_cipher = "".join(list_of_messages)

        if trace > 1:
            print("\ndecrypt - Randomized decrypted cipher: ", randomized_decrypted_cipher, file=trace_f)

        decrypted_message = [str(int(randomized_decrypted_cipher[i])^int(key_for_xor[i]))
                    for i in range(len(randomized_decrypted_cipher))]

        decrypted_message = ''.join(decrypted_message)
        len_decrypted_message = len(decrypted_message)
    
        if trace > 0:
            print("\ndecrypt - Length of decrypted message in bits: {}".format(len_decrypted_message), file=trace_f)
            print("\ndecrypt - Length of decrypted message in bits: {}".format(len_decrypted_message))
        
            elapsed = str(datetime.timedelta(seconds = time.time()-start))
            print("\ndecrypt - Elapsed time for decryption of ciphertext: {}".format(elapsed), file=trace_f)
            print("\ndecrypt - Elapsed time for decryption of ciphertext: {}".format(elapsed))
    
        trace_f.close()
    
        return decrypted_message
    
    ###############################################################################################################
    # Define function file_to_bitstring() to convert plaintext file into a bitstring message                      #
    ###############################################################################################################
    def file_to_bitstring(self):
        
        message_in_bits = None
        
        if self.param == None:
            print("file_to_bitstring: missing param")
            return message_in_bits
        
        if self.json_param == None:
            print("file_to_bitstring: missing self.json_param")
            return message_in_bits
    
        json_param = self.json_param
        param = self.param
    
        QPP_param_file = param['QPP_param_file']
        plaintext_file = param['plaintext_file']
        trace = param['trace']
        trace_file = param['trace_file']
        num_of_qubits = param['num_of_qubits']
    
        trace_f = open(trace_file,"a") # Open trace file in append mode
        
        # Test if plaintext file exists
        if not os.path.isfile(plaintext_file):
            print("\nfile_to_bitstring - Error: missing plaintext file: {}".format(plaintext_file))
            return message_in_bits

        if plaintext_file[-3:] == "png":
            if trace > 0:
                print("file_to_bitstring - Plaintext file {} is an image saved in the Portable Network Graphic (PNG) format".
                      format(plaintext_file), file=trace_f)
    
            # Read image from image file
            from PIL import Image
            from io import BytesIO
            out = BytesIO()
            with Image.open(plaintext_file) as img:
                img.save(out, format="png")
    
            # Convert image to bytes
            message_in_bytes = out.getvalue()
        
            # Get true length of message in bytes
            len_message = len(message_in_bytes)
        
            if trace > 0:
                print("\nfile_to_bitstring - Length of image in bytes: {}".format(len_message), file=trace_f)
                print("\nfile_to_bitstring - Length of image in bytes: {}".format(len_message))
            
                print("\nfile_to_bitstring - First 100 bytes", file=trace_f)
                print(message_in_bytes[:min(len_message, 100)], file=trace_f)
            
                print("\nfile_to_bitstring - Last 100 bytes",file=trace_f)
                print(message_in_bytes[max(len_message-100, 0):len_message], file=trace_f)  
        
            # Convert image in bytes to binary
            message_in_bits = "".join([format(n, '08b') for n in message_in_bytes])
        
            r = len(message_in_bits) % num_of_qubits # Remainder of the length of message divided by the number of qubits
            if r != 0: # If length of message is not a multiple of the number of qubits
                message_in_bits = message_in_bits + '0'*(num_of_qubits - r) # Pad the message with blanks
                print("\nfile_to_bitstring - Length of image in bits padded with '0': {}".format(len(message_in_bits)), file=trace_f) 
        
            msg = " ".join([format(n, '08b') for n in message_in_bytes])
            len_msg = len(msg)
        
            if trace > 0:
                print("\nfile_to_bitstring - First 192 bits in message, shown grouped by 8 bits", file=trace_f)
                print(msg[:min(len_msg, 192)], file=trace_f)
            
                print("\nfile_to_bitstring - Last 192 bits in message, shown grouped by 8 bits", file=trace_f)
                print(msg[max(len_msg-192, 0):len_msg], file=trace_f)

        elif plaintext_file[-3:] == "txt":
            print("file_to_bitstring - Plaintext file {} is a text file".format(plaintext_file), file=trace_f)
    
            # Read text from file
            f = open(plaintext_file, "r")
            plaintext = f.read()
            f.close()
        
            # Get length of message in characters
            len_plaintext = len(plaintext)
        
            if trace > 0:
                print("\nfile_to_bitstring - Length of plain text: {}".format(len_plaintext), file=trace_f)
                print("\nfile_to_bitstring - Length of plain text: {}".format(len_plaintext))
            
                print("\nfile_to_bitstring - First 100 characters in file {}".format(plaintext_file), file=trace_f)
                print(plaintext[:min(len_plaintext, 100)], file=trace_f)
            
                print("\nfile_to_bitstring - Last 100 characters in file {}".format(plaintext_file), file=trace_f)
                print(plaintext[max(len_plaintext-100, 0):len_plaintext], file=trace_f)
            
            # Convert text to bytes
            # The ord() function returns an integer representing the Unicode character.
            message_in_bytes = [ord(i) for i in plaintext]
        
            # Get true length of message in bytes
            len_message = len(message_in_bytes)
        
            if trace > 0:
                print("\nfile_to_bitstring - Length of message in bytes: ", len_message, file=trace_f)
            
                print("\nfile_to_bitstring - First 100 integers representing the Unicode characters with two bytes", file=trace_f)
                print(message_in_bytes[:min(len_message, 100)], file=trace_f)
            
                print("\nfile_to_bitstring - Last 100 integers representing the Unicode characters with two bytes", file=trace_f)
                print(message_in_bytes[max(len_message-100, 0):len_message], file=trace_f)
        
            r = len_message % num_of_qubits # Remainder of the length of plain text divided by the number of qubits
            if r != 0: # If length of plain text is not a multiple of the number of qubits
                plaintext = plaintext + ' '*(num_of_qubits - r) # Pad the plaintext with blanks
                if trace > 0:
                    print("\nfile_to_bitstring - Length of message padded with blanks: ", len(plaintext), file=trace_f)   
    
            # Convert padded text to bytes
            # The ord() function returns an integer representing the Unicode character.
            message_in_bytes = [ord(i) for i in plaintext]
    
            # Convert text in bytes to binary using two bytes for each unicode character
            message_in_bits = "".join([format(n, '016b') for n in message_in_bytes])
            len_message_in_bits = len(message_in_bits)
        
            msg = " ".join([format(n, '016b') for n in message_in_bytes])
            len_msg = len(msg)
        
            if trace > 0:
                print("\nfile_to_bitstring - Length of padded message in bits: ", len_message_in_bits, file=trace_f)
            
                print("\nfile_to_bitstring - First 192 bits in padded message, shown grouped by 16 bits", file=trace_f)
                print(msg[:min(len_msg, 192)], file=trace_f)
            
                print("\nfile_to_bitstring - Last 192 bits in padded message, shown grouped by 16 bits", file=trace_f)
                print(msg[max(len_msg-192, 0):len_msg], file=trace_f)

        else:
            print("file_to_bitstring Error: Unsupported plaintext file: {}".format(plaintext_file), file=trace_f)
            trace_f.close()
            return
       
        # Save true length of message in bytes in param dictionary
        param['len_message'] = len_message
    
        # Save true length of message in bytes in json_param dictionary
        json_param['len_message'] = len_message
    
        # Serializing json
        json_object = json.dumps(json_param, indent=4)
 
        # Write to QPP_param.json
        with open(QPP_param_file, 'w') as json_file:
            json_file.write(json_object)
    
        trace_f.close()
    
        return message_in_bits
    
    ###############################################################################################################
    # Define function ciphertext_to_binary() to convert ciphertext into binary and save it into a binary file     #
    ###############################################################################################################
    def ciphertext_to_binary(self, ciphertext=None):
    
        if ciphertext == None:
            print("ciphertext_to_binary - missing parameter ciphertext")
            return
        
        if self.param == None:
            print("ciphertext_to_binary: missing param")
            return
        
        if self.json_param == None:
            print("ciphertext_to_binary: missing self.json_param")
            return
    
        param = self.param
    
        plaintext_file = param['plaintext_file']
        trace = param['trace']
        trace_file = param['trace_file']
    
        trace_f = open(trace_file,"a") # Open trace file in append mode
    
        # Convert the ciphertext into bytes
        bytes_cipher = bytes(int(ciphertext[i:i+8],2) for i in range(0, len(ciphertext), 8))
    
        # Write the bytes into a binary file
        ciphertext_file = "ciphertext_" + plaintext_file[:-3] + "bin"
        if trace > 0:
            print("ciphertext_to_binary - Ciphertext file name: {}".format(ciphertext_file), file=trace_f)
    
        file_cipher = open(ciphertext_file,"wb")
        file_cipher.write(bytes_cipher)
        file_cipher.close()
    
        trace_f.close()
    
        return
    
    ###############################################################################################################
    # Define function binary_to_ciphertext to read a ciphertext from a binary file                                #
    ###############################################################################################################
    def binary_to_ciphertext(self):
        
        ciphertext = None
        
        if self.param == None:
            print("binary_to_ciphertext: missing param")
            return ciphertext
    
        param = self.param
    
        plaintext_file = param['plaintext_file']
        trace = param['trace']
        trace_file = param['trace_file']
    
        trace_f = open(trace_file,"a") # Open trace file in append mode
    
        ciphertext_file = "ciphertext_" + plaintext_file[:-3] + "bin"
        if trace > 0:
            print("binary_to_ciphertext - Ciphertext file name: {}".format(ciphertext_file), file=trace_f)
    
        # Test if ciphertext file exists
        if os.path.isfile(ciphertext_file):
            file = open(ciphertext_file,"rb")
            a = BitArray(file.read())
            ciphertext = a.bin
        else:
            ciphertext = None
            print("binary_to_ciphertext - Error: missing ciphertext file: {}".format(ciphertext_file), file=trace_f)
    
        trace_f.close()
    
        return ciphertext
    
    ###############################################################################################################
    # Define function bitstring_to_file() to convert a decrypted message and save it into a decrypted file        #
    ###############################################################################################################
    def bitstring_to_file(self, decrypted_message=None):
    
        if decrypted_message == None:
            print("bitstring_to_file - missing parameter decrypted_message")
            return
        
        if self.param == None:
            print("bitstring_to_file: missing param")
            return
    
        param = self.param
    
        plaintext_file = param['plaintext_file']
        trace = param['trace']
        trace_file = param['trace_file']
    
        trace_f = open(trace_file,"a") # Open trace file in append mode
    
        # Create a decrypted file
        decrypted_file = "Decrypted_" + plaintext_file
        print("bitstring_to_file - Decrypted file name: {}".format(decrypted_file), file=trace_f)
    
        # Convert the decrypted message and save it into the decrypted file
        if decrypted_file[-3:] == "png": 
        
            # Convert the decrypted message into bytes
            decrypted_bytes = [int(decrypted_message[i: i+8], 2) for i in range(0, len(decrypted_message), 8)]
        
            # Trim decrypted message to the true length
            decrypted_bytes = decrypted_bytes[:param['len_message']]
            len_decrypted_bytes = len(decrypted_bytes)
        
            if trace > 0:
                print("\nbitstring_to_file - decrypted message in bytes trimmed to true length: {}".format(param['len_message']), 
                      file=trace_f)
                print("\nbitstring_to_file - decrypted message in bytes trimmed to true length: {}".format(param['len_message']))
        
            msg = " ".join([format(n, '08b') for n in decrypted_bytes])
            len_msg = len(msg)
        
            if trace > 0:
                print("\nbitstring_to_file - Length of image in bytes: {}".format(len_decrypted_bytes), file=trace_f)
            
                print("\nbitstring_to_file - First 192 bits in decrypted message, shown grouped by 8 bits", file=trace_f)
                print(msg[:min(len_msg, 192)], file=trace_f)
            
                print("\nbitstring_to_file - Last 192 bits in decrypted message, shown grouped by 8 bits", file=trace_f)
                print(msg[max(len_msg-192, 0):len_msg], file=trace_f)
        
            # Write decrypted bytes in file
            with open(decrypted_file, 'wb') as f:
                f.write(bytes(decrypted_bytes))
            f.close()

        elif decrypted_file[-3:] == "txt":
            # Convert the decrypted message into bytes, using two bytes per unicode character
            decrypted_bytes = [int(decrypted_message[i: i+16], 2) for i in range(0, len(decrypted_message), 16)]
        
            # Trim decrypted message to the true length
            decrypted_bytes = decrypted_bytes[:param['len_message']]
        
            if trace > 0:
                print("\nbitstring_to_file - decrypted message in bytes trimmed to true length {}".format(param['len_message']), 
                      file=trace_f)
                print("\nbitstring_to_file - decrypted message in bytes trimmed to true length {}".format(param['len_message']))
        
            msg = " ".join([format(n, '016b') for n in decrypted_bytes])
            len_msg = len(msg)
        
            # Convert the decrypted message from bytes to a string
            s = "".join(map(chr, decrypted_bytes))
            len_s = len(s)
        
            if trace > 0:
                print("\nbitstring_to_file - First 192 bits in message, shown grouped by 16 bits", file=trace_f)
                print(msg[:min(len_msg, 192)], file=trace_f)
            
                print("\nbitstring_to_file - First 100 decrypted integers representing the Unicode characters with two bytes", file=trace_f)
                print(decrypted_bytes[:min(len_s, 100)], file=trace_f)
            
                print("\nbitstring_to_file - First 100 decrypted characters in file {}".format(decrypted_file), file=trace_f)
                print(s[:min(len_s, 100)], file=trace_f)
            
                print("\nbitstring_to_file - Last 192 bits in message, shown grouped by 16 bits", file=trace_f)
                print(msg[max(len_msg-192, 0):len_msg], file=trace_f)
            
                print("\nbitstring_to_file - Last 100 decrypted integers representing the Unicode characters with two bytes", file=trace_f)
                print(decrypted_bytes[max(len_s-100, 0):len_s], file=trace_f)
            
                print("\nbitstring_to_file - Last 100 decrypted characters in file {}".format(decrypted_file), file=trace_f)
                print(s[max(len_s-100, 0):len_s], file=trace_f)
    
            # Write decrypted bytes in file
            f = open(decrypted_file, "w")
            f.write(s)
            f.close()

        else:
            print("bitstring_to_file - Unsupported decrypted file: {}".format(decrypted_file), file=trace_f)
    
        trace_f.close()
    
        return
