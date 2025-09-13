# run_exit_list.py
import subprocess
import numpy as np

# List of Python scripts with arguments to run
L0s = np.arange(8, 8 + 1)
scripts = [("run_exit.py", str(L0)) for L0 in L0s]

for script, arg in scripts:
    print(f"Running {script} {arg}...")
    try:
        subprocess.run(["python", script, arg], check=True)
        print(f"{script} {arg} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script} {arg}: {e}")
        break
