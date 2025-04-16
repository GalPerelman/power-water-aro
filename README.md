# power-water-aro
Adjustable Robust Optimization for Integrated Power and Water Distribution Systems Under Uncertainty</br>

### Abstract
Water distribution systems and power distribution systems are strongly connected and mutually affect each other. Recently, several studies have suggested leveraging this dependency to enhance the overall efficiency of these systems. However, the uncertainty inherent in the operation of such systems got less attention. In this study, Robust Optimization (RO) and Adjustable Robust Optimization (ARO) are suggested to co-optimize the day-ahead scheduling of generators dispatch and pump operation while considering several sources of uncertainties in both systems. The paper presents a novel formulation for modeling power balance and mass balance equality constraints, which hinder the implementation of RO and ARO. Numerical results on two case studies demonstrate the pareto tradeoff between optimality and robustness and the superiority of ARO due to its ability to adapt to real-time dynamics in the networks

### Installation
First, initiate a Python virtual environment
Open a terminal window in the project directory
Create new venv by running the command:</br>
`python -m venv <venv name>`</br>
Activate the venv by:</br>
`source venv/bin/activate` (for mac) or `venv\Scripts\activate.bat` (for windows)</br>

Once the virtual environment is set up and activated,</br>
Install the dependencies by running the following command:
`pip install -r requirements.txt`</br>

### Run the code
To run an experiment, select a configuration file from the experiments directory and run the following:</br>
`python main.py --config_path experiments/I_3-bus_desalination_wds_aro.yaml --command experiment`</br>
To run a latency analysis, run the following:</br>
`python main.py --config_path experiments/I_3-bus_desalination_wds_aro.yaml --command latency --omega 1 --pds_lat 0 2 4 --wds_lat 4 6 8`

The output will be saved in the `output` directory
To generate the paper figures based on the results in the output directory run the `simulation.py` file