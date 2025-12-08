# QPP-Alain
QPP implementations from 2 to 9 qubits with Gradio UI and a JSON-RPC 2.0 file exchange.

This repository contains Jupyter notebooks for Quantum Permutation Pad (QPP) implementations with Qiskit Runtime, Qiskit AerSimulator and Perceval, Quandela's Python framework for designing and simulating photonic quantum circuits.  They serve as illustrations for the paper: Chancé, A. (2024). Quantum Permutation Pad with Qiskit Runtime. In: Femmam, S., Lorenz, P. (eds). ICCNT 2022. Springer, Cham. https://doi.org/10.1007/978-3-031-59619-3_12 .
Implementations with Qiskit range from 2 to 9 qubits, and implementations with Perceval cover permutation sizes from 2 to 8.

🎤 Alain's talk, “An Efficient 8-Qubit Quantum Permutation Pad (QPP) Implementation Running on a Laptop”, has been presented at the 2025 IEEE 14th International Conference on Communications, Circuits and Systems (ICCCAS).

It is part of Invited Speech Session C: Quantum Science and Technology, taking place on Sunday, May 25, 2025, from 13:00–15:15 CST (Beijing) / 07:00–09:15 CEST (Paris). Chair: Randy Kuang, Quantropi Inc., Ottawa, Canada & Michel Barbeau, Carleton University, Canada.
📄 Conference website: www.icccas.org. Conference Programme: https://www.icccas.org/prog.html

![Certificate of achievement](https://github.com/AlainChance/QPP-Alain/blob/main/Certificate%20of%20achievement.png)

Check out a recording of Alain's speech at the 12th Edition of the ICCCAS in Singapore, May 5–7, 2023, 
- on YouTube: https://www.youtube.com/watch?v=aUnWFgz-vio
- or on Bilibili: https://www.bilibili.com/video/BV1rw411r7jR/

Randy Kuang's [LinkedIn profile](https://www.linkedin.com/in/randy-kuang-05150510/)

Michel Barbeau's [LinkedIn profile](https://www.linkedin.com/in/michel-barbeau-464906291/) and https://carleton.ca/scs/people/michel-barbeau/

## Set up and save your IBM Cloud account on the upgraded IBM Quantum Platform (optional) 
Refer to [this guide](https://quantum.cloud.ibm.com/docs/en/guides/cloud-setup) for details on how to set up your IBM Cloud account on the upgraded IBM Quantum Platform.

In order to run [QPP_Alice.ipynb](https://github.com/AlainChance/QPP-Alain/blob/main/Alice/QPP_Alice.ipynb) and [QPP_Bob.ipynb](https://github.com/AlainChance/QPP-Alain/blob/main/Bob/QPP_Bob.ipynb) on the upgraded IBM Quantum Platform, either save your credentials before running them or copy your API token in a file named "Token.txt", copy the CRN code associated with your instance in a file named "CRN.txt" and then copy both files in the directory that contains the jupyter notebooks.

## 1. Run [Bob_agent.ipynb](https://github.com/AlainChance/QPP-Alain/blob/main/Bob/Bob_agent.ipynb) 
It starts a receiver agent which functions as a uvicorn server and receives a file using:
* 🛰️ FastAPI on the server (Remote Agent)
* 📡 requests module on the client
* 📦 File content encoded in Base64
* 🌐 JSON-RPC 2.0 over HTTP

## 2. Run [QPP_Alice.ipynb](https://github.com/AlainChance/QPP-Alain/blob/main/Alice/QPP_Alice.ipynb)
When the receiver agent is not reachable, the following prompt is displayed:
- `Start a receiver agent with Bob_agent.ipynb - and enter 'y' when done`

Answer the following prompt:
- `Start Gradio UI interface? (y/n)`

### QPP setup with Gradio UI
The following message is printed:
- `Running Gradio UI on local URL: http://127.0.0.1:7860`

Open URL `http://127.0.0.1:7860` and set up a configuration, see the examples shown below:

![QPP_config_and_run.png](https://github.com/AlainChance/QPP-Alain/blob/main/QPP_2_qubits/le-mont-saint-michel-normandy-france/Alice/AerSimulator%20noiseless/QPP_config_run.png)
![QPP_config_and_run_SLOS.png](https://github.com/AlainChance/QPP-Alain/blob/main/QPP_8_qubits/le-mont-saint-michel-normandy-france/Alice/SLOS/QPP_config_run_SLOS.png)

Complete the QPP setup and answer the following prompts:
- `QPP configuration with Gradio complete? (y/n)`
- `Run QPP simulation? (y/n)`

### QPP setup with prompts 
1/ Answer the following prompts:
- `Enter plaintext filename (.txt or .png or .jpg)`
- `Enter number of qubits between 2 and 9`
- `Enter version 0 (n qubits) or 1 (2**n qubits which only uses swap gates)`
- `Enter trace level 0 or 1`
- `Enter 0 for classical simulation or 1 for running on a real QPU or a simulation with AerSimulator`
- `Enter IBM cloud backend name or 'None' to assign least busy device to backend or 'AerSimulator noiseless'`

If the answer is in `['None', 'ibm_sherbrooke', 'ibm_brisbane', 'ibm_torino']` then answer the following prompt:
- `Enter maximum number of permutations in pad between 0 (no maximum) and 56`.

### QPP simulation
The secret key, QPP parameters and ciphertext files are sent to Bob receiver agent with the send_file method using:
* 📡 requests module
* 📦 File content encoded in Base64
* 🌐 JSON-RPC 2.0 over HTTP

### Saving the content of the Alice directory
Following the message in bold: "Do a File Save of this notebook":
- `Do you want to save the content of the Alice directory? (y/n)`
If the answer is "y" then Alice directory is copied into a new directory. 

## 3. Run [QPP_Bob.ipynb](https://github.com/AlainChance/QPP-Alain/blob/main/Bob/QPP_Bob.ipynb)
It does the following: 

1/ Prompt
- `Enter version 0 (n qubits) or 1 (2**n qubits which only uses swap gates)`

2/ Decrypt the text or the image file received from Alice using the Json parameter file and the secret key file.

3/ Print in bold `Do a File Save of this notebook` and then prompts:
- `Do you want to save the content of the Bob directory? (y/n)`
If the answer is "y" then copy Bob directory into a new one. 

## Compatible Qiskit versions
These Jupyter notebooks work with Python 3.13 and the following Qiskit versions:
- Qiskit v2.1, Qiskit runtime 0.40, Qiskit Aer 0.17

Open Plan users cannot submit session jobs. Workloads must be run in job mode or batch mode, https://quantum.cloud.ibm.com/docs/en/guides/run-jobs-session

Please refer to the Qiskit documentation in section References below.

## License, abstract, credits, rights and permissions
The first cell of QPP_Alice.ipynb and QPP_Bob.ipynb notebooks contains the following: 
- MIT license
- Abstract
- Credit: Kuang, R., Perepechaenko
- Rights and permissions
- Adaptations made by Alain Chancé

Updates are presented in the following cells at the end of these notebooks:
- Summary of updates V6
- Summary of updates V5
- Summary of updates V4
- Summary of updates V3
- Summary of updates V2
- Summary of updates V1      

## References

[1] Kuang, Randy. Quantum Permutation Pad for Quantum Secure Symmetric and Asymmetric Cryptography. Vol. 2, no. 1, Academia Quantum, 2025. https://doi.org/10.20935/AcadQuant7457

[2] I. Burge, M. T. Mai and M. Barbeau, "A Permutation Dispatch Circuit Design for Quantum Permutation Pad Symmetric Encryption," 2024 13th International Conference on Communications, Circuits and Systems (ICCCAS), Xiamen, China, 2024, pp. 35-40, doi: 10.1109/ICCCAS62034.2024.10652827.

[3] Chancé, A. (2024). Quantum Permutation Pad with Qiskit Runtime. In: Femmam, S., Lorenz, P. (eds) Recent Advances in Communication Networks and Embedded Systems. ICCNT 2022. Lecture Notes on Data Engineering and Communications Technologies, vol 205. Springer, Cham. https://doi.org/10.1007/978-3-031-59619-3_12

[4] Kuang, R., Barbeau, M. Quantum permutation pad for universal quantum-safe cryptography. Quantum Inf Process 21, 211 (2022). https://doi.org/10.1007/s11128-022-03557-y

[5] R. Kuang and N. Bettenburg, 'Shannon perfect secrecy in a discrete Hilbert space', in Proc. IEEE Int. Conf. Quantum Comput. Eng. (QCE), Oct. 2020, pp. 249-255, doi: 10.1109/QCE49297.2020.00039

[6] Kuang, R., Perepechaenko, M. Quantum encryption with quantum permutation pad in IBMQ systems. EPJ Quantum Technol. 9, 26 (2022). https://doi.org/10.1140/epjqt/s40507-022-00145-y

[7] Qiskit Runtime overview, IBM Quantum, https://cloud.ibm.com/docs/quantum-computing?topic=quantum-computing-overview

[8] QiskitRuntimeService, https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.QiskitRuntimeService#qiskitruntimeservice

[9] Qiskit v2.0 migration guide, https://docs.quantum.ibm.com/migration-guides/qiskit-2.0

[10] Qiskit Aer documentation, https://qiskit.github.io/qiskit-aer/

[11] Qiskit Aer 0.16.1, Getting started, https://qiskit.github.io/qiskit-aer/getting_started.html

[12] Qiskit Aer 0.16.1, Simulators, https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html

[13] Migrate from cloud simulators to local simulators, https://docs.quantum.ibm.com/migration-guides/local-simulators#aersimulator
