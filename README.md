# Solar Modeling

A tool to model cost and utilization of solar power systems. Uses power generation data from [NREL's 2006 solar plant datasets](https://www.nrel.gov/grid/solar-power-data).

https://alecchen.dev/images/solar_demo.mp4

(optimization step count decreased for demo)

### Usage

*See possible improvements? File an issue! Feedback is welcome!*

- Create virtual environment
    - `python3 -m venv venv`
    - `source venv/bin/activate` or whichever works for your shell, e.g. `source venv/bin/activate.fish`
- Install dependencies
    - `python -m pip install -r requirements.txt`
- Run script
    - `python solar.py`
    - Note: after first installing dependencies, the script may take a while to start. Subsequent runs should be quick.
- Run tests
    - `python test.py`
- Type check
    - `mypy solar.py test.py`
- Format/lint
    - `ruff format`
    - `ruff check`

Notes on specific commands
- For commands where you compute the optimal power configuration (`optimize` and `all`), the default is to run
for 41 different load costs. This will take roughly 9 minutes for a plant with 5 min
intervals. You can change this by modifying the constant `DEFAULT_LOAD_COSTS` in the
code (see To Do section for making this better).
- For creating the `states_power_cost_per_energy_by_load_cost.png` graph, currently
the only way to do this is to run the `all` command on the states you want to compare (see To Do section for making this better).

### Goals
- Prescribe a way to do things to simplify usage, but allow user control
- Be able to run one command for installation, one command to run the script
- Script should be able to go from nothing downloaded, all the way to the final results in one command
- Script should be able to take any intermediate input along the way and produce the next intermediate output
- Script should be easy to use as a library if imported, but also easy to copy paste chunks and modify, say in a jupyter notebook
- Script should be a single file (we're not doing anything super complicated here, everything should fit comfortably in one file)
- If a chunk takes time to process, we should print something to inform the user
- Make things simple

### To do
- Plots
    - Create cost per usage by load cost graph comparing locations by inputting different state files instead of needing to run the `all` command
    - Plant stats (e.g. histogram of capacities for a state)
    - Utilization for each day over a year
    - Power generation overlaying days for a full year
- Optimize - allow specifying load costs, or at least a test flag to only optimize over a single load cost
- Combine east/west dataset separations (Texas, New Mexico only)
- Code - delete StrEnums for column name constants?
- Delete mean_plant_for_state, just choose one deterministically
- Pre-compute optimal results/visuals and serve them from a website so people
only need to run this script if they want to verify or customize
