# Setup instructions

1. Install [Python 3](https://www.python.org/downloads/). On macOS you can do it with homebrew:
```bash
brew install python
```

2. Navigate to the project's directory:
```bash
cd anthroproject
```

3. Set up your Python virtual environment:
```bash
python -m venv ./env-anthroproject
```

This creates a folder for your virtual environment `env-anthroproject` where all the required packages are saved.

4. Activate the virtual environment. Here is how you to it on Windows:
```bash
.\env-anthroproject\Scripts\activate
```

Here is how you do it on macOS:
```bash
source env-anthroproject/bin/activate
```

5. Install the required Python packages (in the activated environment):

```bash
pip install -r requirements.txt
```

6. Put your `credentials` file in the folder.

7. Run the Python script 
```bash
python json2csv.py
```

The script downloads the AWS raw JSON data files, executes transformations, and outputs into CSV files

The JSON files are stored in `data/json/` and the resulting CSV files are stored in `data/csv/`
