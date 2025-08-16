# Solar Modeling

Usage
- Create virtual environment
    - `python3 -m venv venv`
    - `source venv/bin/activate` or whichever works for your shell, e.g. `source venv/bin/activate.fish`
- Install dependencies
    - `pip install -r requirements.txt`
- Run script
    - `python3 solar.py`
- Run tests
    - `python3 test.py`

Goals
- Prescribe a way to do things to simplify usage, but allow multiple options
- Be able to run one command for installation, one command to run the script
- Script should be able to go from nothing downloaded, all the way to the final results in one command
- Script should be able to take any intermediate input along the way and produce the next intermediate output
- Script should be easy to use as a library if imported, but also easy to copy paste chunks and modify, say in a jupyter notebook
- Script should be a single file (we're not doing anything super complicated here, everything should fit in one file)
- If a chunk takes time to process, we should print something to inform the user

To do
- Downloading
    - [ ] Give options to just read a cached file of the datasets, vs. crafting one from all the files
    - [ ] Make another repo to save all the zips - script lets you choose whether to download all, download one/some, unzip or read from local
