import sys
import subprocess
import time
import tkinter as tk
from tkinter import messagebox
import importlib.util
from pathlib import Path

# Define path to your main script directory
MAIN_SCRIPT_DIR = Path(r"D:\Scripts\VectorDB-Plugin-for-LM-Studio\working")

# Import modules from the main script directory
def import_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import necessary modules
constants = import_module("constants", MAIN_SCRIPT_DIR / "constants.py")
replace_sourcecode = import_module("replace_sourcecode", MAIN_SCRIPT_DIR / "replace_sourcecode.py")

start_time = time.time()

def has_nvidia_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
hardware_type = "GPU" if has_nvidia_gpu() else "CPU"

def tkinter_message_box(title, message, type="info", yes_no=False):
    root = tk.Tk()
    root.withdraw()
    if yes_no:
        result = messagebox.askyesno(title, message)
    elif type == "error":
        messagebox.showerror(title, message)
        result = False
    else:
        messagebox.showinfo(title, message)
        result = True
    root.destroy()
    return result

def check_python_version_and_confirm():
    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if major == 3 and minor in [11, 12]:
        return tkinter_message_box("Confirmation", f"Python version {sys.version.split()[0]} was detected, which is compatible.\n\nClick YES to proceed or NO to exit.", type="yesno", yes_no=True)
    else:
        tkinter_message_box("Python Version Error", "This program requires Python 3.11 or 3.12\n\nPython versions prior to 3.11 or after 3.12 are not supported.\n\nExiting the installer...", type="error")
        return False

def is_nvidia_gpu_installed():
    try:
        subprocess.check_output(["nvidia-smi"])
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def manual_installation_confirmation():
    if not tkinter_message_box("Confirmation", "Have you installed Git?\n\nClick YES to confirm or NO to cancel installation.", type="yesno", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", "Have you installed Git Large File Storage?\n\nClick YES to confirm or NO to cancel installation.", type="yesno", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", "Have you installed Pandoc?\n\nClick YES to confirm or NO to cancel installation.", type="yesno", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", "Have you installed Microsoft Build Tools and/or Visual Studio with the necessary libraries to compile code?\n\nClick YES to confirm or NO to cancel installation.", type="yesno", yes_no=True):
        return False
    return True

if not check_python_version_and_confirm():
    sys.exit(1)

nvidia_gpu_detected = is_nvidia_gpu_installed()
if nvidia_gpu_detected:
    message = "An NVIDIA GPU has been detected.\n\nDo you want to proceed with the installation?"
else:
    message = "No NVIDIA GPU has been detected. An NVIDIA GPU is required for this script to function properly.\n\nDo you still want to proceed with the installation?"

if not tkinter_message_box("GPU Detection", message, type="yesno", yes_no=True):
    sys.exit(1)

if not manual_installation_confirmation():
    sys.exit(1)

def upgrade_pip_setuptools_wheel(max_retries=5, delay=3):
    upgrade_commands = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--no-cache-dir"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "setuptools", "--no-cache-dir"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "wheel", "--no-cache-dir"]
    ]
    
    for command in upgrade_commands:
        package = command[5]
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1} of {max_retries}: Upgrading {package}...")
                process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=480)
                print(f"\033[92mSuccessfully upgraded {package}\033[0m")
                break
            except subprocess.CalledProcessError as e:
                print(f"Attempt {attempt + 1} failed. Error: {e.stderr.strip()}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Failed to upgrade {package} after {max_retries} attempts.")
            except Exception as e:
                print(f"An unexpected error occurred while upgrading {package}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Failed to upgrade {package} after {max_retries} attempts due to unexpected errors.")

def pip_install_with_retry(library, max_retries=5, delay=3):
    if "git+" in library:
        pip_args_list = [["uv", "pip", "install", library, "--no-deps"]]
    else:
        pip_args_list = [["uv", "pip", "install", library, "--no-deps"]]
    
    for pip_args in pip_args_list:
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1} of {max_retries}: Installing {pip_args[3]}")
                subprocess.run(pip_args, check=True, capture_output=True, text=True, timeout=480)
                print(f"\033[92mSuccessfully installed {pip_args[3]}\033[0m")
                break
            except subprocess.CalledProcessError as e:
                print(f"Attempt {attempt + 1} failed. Error: {e.stderr.strip()}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Failed to install {pip_args[3]} after {max_retries} attempts.")
                    return 0
    return 1

def install_libraries(libraries):
    failed_installations = []
    multiple_attempts = []

    for library in libraries:
        attempts = pip_install_with_retry(library)
        if attempts == 0:
            failed_installations.append(library)
        elif attempts > 1:
            multiple_attempts.append((library, attempts))
        time.sleep(0.1)

    return failed_installations, multiple_attempts

def pip_install_with_deps(library, max_retries=5, delay=3):
    pip_args = ["uv", "pip", "install", library]

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}: Installing {library} with dependencies")
            subprocess.run(pip_args, check=True, capture_output=True, text=True, timeout=600)
            print(f"\033[92mSuccessfully installed {library} with dependencies\033[0m")
            return attempt + 1
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed. Error: {e.stderr.strip()}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to install {library} after {max_retries} attempts.")
                return 0

def install_libraries_with_deps(libraries):
    failed_installations = []
    multiple_attempts = []

    for library in libraries:
        attempts = pip_install_with_deps(library)
        if attempts == 0:
            failed_installations.append(library)
        elif attempts > 1:
            multiple_attempts.append((library, attempts))
        time.sleep(0.1)

    return failed_installations, multiple_attempts

# 1. Upgrade pip, setuptools, wheel
print("Upgrading pip, setuptools, and wheel:")
upgrade_pip_setuptools_wheel()

# 2. Install uv
print("Installing uv:")
subprocess.run(["pip", "install", "uv"], check=True)

# 3. Install priority_libs
print("\nInstalling priority libraries:")
try:
    hardware_specific_libs = constants.priority_libs[python_version][hardware_type]
    
    # Correctly handle COMMON libraries
    try:
        common_libs = constants.priority_libs[python_version]["COMMON"]
    except KeyError:
        common_libs = []
    
    all_priority_libs = hardware_specific_libs + common_libs
    
    priority_failed, priority_multiple = install_libraries(all_priority_libs)
except KeyError:
    tkinter_message_box("Version Error", f"No libraries configured for Python {python_version} with {hardware_type} configuration", type="error")
    sys.exit(1)

# 4. Install libs
print("\nInstalling other libraries:")
other_failed, other_multiple = install_libraries(constants.libs)

# 5. Install full_install_libs
print("\nInstalling libraries with dependencies:")
full_install_failed, full_install_multiple = install_libraries_with_deps(constants.full_install_libs)

print("\n----- Installation Summary -----")

all_failed = priority_failed + other_failed + full_install_failed
all_multiple = priority_multiple + other_multiple + full_install_multiple

if all_failed:
    print("\033[91m\nThe following libraries failed to install:\033[0m")
    for lib in all_failed:
        print(f"\033[91m- {lib}\033[0m")

if all_multiple:
    print("\033[93m\nThe following libraries required multiple attempts to install:\033[0m")
    for lib, attempts in all_multiple:
        print(f"\033[93m- {lib} (took {attempts} attempts)\033[0m")

if not all_failed and not all_multiple:
    print("\033[92mAll libraries installed successfully on the first attempt.\033[0m")
elif not all_failed:
    print("\033[92mAll libraries were eventually installed successfully.\033[0m")

if all_failed:
    sys.exit(1)

# Replace source code files - using imported functions from replace_sourcecode
replace_sourcecode.replace_sentence_transformer_file()
replace_sourcecode.replace_chattts_file() 
replace_sourcecode.add_cuda_files()

# Note: We're intentionally skipping these functions per your requirements:
# - replace_sourcecode.setup_vector_db()
# - replace_sourcecode.check_embedding_model_dimensions()
# - clean_triton_cache()
# - create_directory_structure() 
# - update_config_yaml()

end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"\033[92m\nTotal installation time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\033[0m")