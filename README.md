# Solar Modeling

A tool to model cost and utilization of solar power systems. Uses power generation data from [NREL's 2006 solar plant datasets](https://www.nrel.gov/grid/solar-power-data).

### Usage
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

### Goals
- Prescribe a way to do things to simplify usage, but allow user control
- Be able to run one command for installation, one command to run the script
- Script should be able to go from nothing downloaded, all the way to the final results in one command
- Script should be able to take any intermediate input along the way and produce the next intermediate output
- Script should be easy to use as a library if imported, but also easy to copy paste chunks and modify, say in a jupyter notebook
- Script should be a single file (we're not doing anything super complicated here, everything should fit comfortably in one file)
- If a chunk takes time to process, we should print something to inform the user

### To do
- Plots
    - Plant stats (e.g. histogram of capacities for a state)
    - Utilization for each day over a year
    - Power generation overlaying days for a full year
- Optimize - allow specifying load costs, or at least a test flag to only optimize over a single load cost
- Code - delete StrEnums for column name constants?
