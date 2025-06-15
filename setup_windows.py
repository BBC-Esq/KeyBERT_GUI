import os
import subprocess
import sys
import time
import tkinter as tk
from tkinter import messagebox

cache_dir = os.path.join(
    os.environ.get("USERPROFILE", os.path.expanduser("~")),
    ".triton"
)

if os.path.isdir(cache_dir):
    print(f"\nRemoving Triton cache at {cache_dir} via OS command…")
    subprocess.run(f'rmdir /S /Q "{cache_dir}"', shell=True, check=False)
    print("Triton cache removed.\n")
else:
    print("\nNo Triton cache found to clean.\n")

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


priority_libs = {
    "cp310": {
        "CPU": [
            "https://download.pytorch.org/whl/cpu/torch-2.2.2%2Bcpu-cp310-cp310-win_amd64.whl"
        ],
        "GPU": [
            "https://download.pytorch.org/whl/cu121/torch-2.2.2%2Bcu121-cp310-cp310-win_amd64.whl",
            "nvidia-cuda-runtime-cu12==12.4.127",
            "nvidia-cublas-cu12==12.4.5.8",
            "nvidia-cuda-nvrtc-cu12==12.4.127",
            "nvidia-cudnn-cu12==9.1.0.70"
        ],
        "COMMON": []
    },
    "cp311": {
        "CPU": [
            "https://download.pytorch.org/whl/cpu/torch-2.2.2%2Bcpu-cp311-cp311-win_amd64.whl"
        ],
        "GPU": [
            "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-win_amd64.whl",
            "https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp311-cp311-win_amd64.whl",
            "https://download.pytorch.org/whl/cu124/torchaudio-2.6.0%2Bcu124-cp311-cp311-win_amd64.whl",
            "triton-windows==3.2.0.post18",
            "xformers==0.0.29.post3",
            "nvidia-cuda-runtime-cu12==12.4.127",
            "nvidia-cublas-cu12==12.4.5.8",
            "nvidia-cuda-nvrtc-cu12==12.4.127",
            "nvidia-cuda-nvcc-cu12==12.4.131",
            "nvidia-cufft-cu12==11.2.1.3",
            "nvidia-cudnn-cu12==9.1.0.70",
            "nvidia-ml-py==12.575.51"
        ],
        "COMMON": []
    },
    "cp312": {
        "CPU": [
        ],
        "GPU": [
            "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl",
            "https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp312-cp312-win_amd64.whl",
            "https://download.pytorch.org/whl/cu124/torchaudio-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl",
            "triton-windows==3.2.0.post18",
            "xformers==0.0.29.post3",
            "nvidia-cuda-runtime-cu12==12.4.127",
            "nvidia-cublas-cu12==12.4.5.8",
            "nvidia-cuda-nvrtc-cu12==12.4.127",
            "nvidia-cuda-nvcc-cu12==12.4.131",
            "nvidia-cufft-cu12==11.2.1.3",
            "nvidia-cudnn-cu12==9.1.0.70",
            "nvidia-ml-py==12.575.51"
        ],
        "COMMON": []
    }
}

libs = [
    "annotated-types==0.7.0",
    "blis==0.7.11",
    "catalogue==2.0.10",
    "certifi==2025.6.15",
    "charset-normalizer==3.4.2",
    "click==8.1.8",
    "cloudpathlib==0.19.0",
    "colorama==0.4.6",
    "confection==0.1.5",
    "cymem==2.0.8",
    "filelock==3.18.0",
    "fsspec==2024.9.0",
    "huggingface-hub==0.33.0",
    "idna==3.10",
    "Jinja2==3.1.6",
    "joblib==1.5.1",
    "langcodes==3.4.1",
    "language_data==1.2.0",
    "marisa-trie==1.2.0",
    "markdown-it-py==3.0.0",
    "MarkupSafe==3.0.2",
    "mdurl==0.1.2",
    "mpmath==1.3.0",
    "murmurhash==1.0.10",
    "networkx==3.5",
    "numpy==2.2.6",
    "packaging==24.2",
    "pillow==11.2.1",
    "preshed==3.0.9",
    "pydantic==2.11.7",
    "pydantic_core==2.33.2",
    "Pygments==2.19.1",
    "PyYAML==6.0.2",
    "regex==2024.11.6",
    "requests==2.32.4",
    "rich==14.0.0",
    "safetensors==0.5.3",
    "scikit-learn==1.7.0",
    "scipy==1.15.3",
    "shellingham==1.5.4",
    "shiboken6==6.7.3",
    "smart-open==7.0.5",
    "spacy-legacy==3.0.12",
    "spacy-loggers==1.0.5",
    "srsly==2.4.8",
    "sympy==1.13.1",
    "thinc==8.2.5",
    "threadpoolctl==3.6.0",
    "tokenizers==0.21.1",
    "tqdm==4.67.1",
    "transformers==4.52.4",
    "typer==0.12.5",
    "typing_extensions==4.14.0",
    "urllib3==2.4.0",
    "wasabi==1.1.3",
    "weasel==0.4.1",
    "wrapt==1.17.2"
]

full_install_libs = [
    "keybert==0.8.5",
    "sentence-transformers==4.1.0",
    "spacy==3.7.5",
    "PySide6==6.9.1"
]


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
    if major == 3 and minor in [10, 11, 12]:
        return tkinter_message_box(
            "Confirmation",
            f"Python version {sys.version.split()[0]} was detected, which is compatible.\n\nClick YES to proceed or NO to exit.",
            yes_no=True
        )
    else:
        tkinter_message_box(
            "Python Version Error",
            "This program requires Python 3.10, 3.11 or 3.12\n\nOther Python versions are not supported.\n\nExiting the installer...",
            type="error"
        )
        return False

def is_nvidia_gpu_installed():
    try:
        subprocess.check_output(["nvidia-smi"])
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def manual_installation_confirmation():
    if not tkinter_message_box("Confirmation", "Have you installed Git?\n\nClick YES to confirm or NO to cancel installation.", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", "Have you installed Microsoft Build Tools and/or Visual Studio with the necessary libraries to compile code?\n\nClick YES to confirm or NO to cancel installation.", yes_no=True):
        return False
    return True

if not check_python_version_and_confirm():
    sys.exit(1)

nvidia_gpu_detected = is_nvidia_gpu_installed()
if nvidia_gpu_detected:
    message = "An NVIDIA GPU has been detected.\n\nDo you want to install the GPU-accelerated version?"
else:
    message = "No NVIDIA GPU has been detected. The CPU-only version will be installed.\n\nDo you want to proceed?"

if not tkinter_message_box("GPU Detection", message, yes_no=True):
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
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=480)
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

def pip_install(library, with_deps=False, max_retries=5, delay=3):
    pip_args = ["uv", "pip", "install", library]
    if not with_deps:
        pip_args.append("--no-deps")

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}: Installing {library}{' with dependencies' if with_deps else ''}")
            subprocess.run(pip_args, check=True, capture_output=True, text=True, timeout=600)
            print(f"\033[92mSuccessfully installed {library}{' with dependencies' if with_deps else ''}\033[0m")
            return attempt + 1
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed. Error: {e.stderr.strip()}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to install {library} after {max_retries} attempts.")
                return 0

def install_libraries(libraries):
    failed_installations = []
    multiple_attempts = []

    for library in libraries:
        attempts = pip_install(library)
        if attempts == 0:
            failed_installations.append(library)
        elif attempts > 1:
            multiple_attempts.append((library, attempts))
        time.sleep(0.1)

    return failed_installations, multiple_attempts

def install_libraries_with_deps(libraries):
    failed_installations = []
    multiple_attempts = []

    for library in libraries:
        attempts = pip_install(library, with_deps=True)
        if attempts == 0:
            failed_installations.append(library)
        elif attempts > 1:
            multiple_attempts.append((library, attempts))
        time.sleep(0.1)

    return failed_installations, multiple_attempts

def clean_triton_cache():
    cache_dir = os.path.join(
        os.environ.get("USERPROFILE", os.path.expanduser("~")),
        ".triton"
    )
    if os.path.isdir(cache_dir):
        print(f"\nCleaning Triton cache at {cache_dir}...")
        subprocess.run(f'rmdir /S /Q "{cache_dir}"', shell=True, check=False)
        print("Triton cache cleaned.\n")

# Main installation process
print("Upgrading pip, setuptools, and wheel:")
upgrade_pip_setuptools_wheel()

print("Installing uv:")
subprocess.run(["pip", "install", "uv"], check=True)

print("\nInstalling priority libraries (PyTorch):")
try:
    hardware_specific_libs = priority_libs[python_version][hardware_type]
    
    try:
        common_libs = priority_libs[python_version]["COMMON"]
    except KeyError:
        common_libs = []

    all_priority_libs = hardware_specific_libs + common_libs
    priority_failed, priority_multiple = install_libraries(all_priority_libs)
except KeyError:
    tkinter_message_box("Version Error", f"No libraries configured for Python {python_version} with {hardware_type} configuration", type="error")
    sys.exit(1)

print("\nInstalling core libraries:")
other_failed, other_multiple = install_libraries(libs)

print("\nInstalling libraries with dependencies:")
full_install_failed, full_install_multiple = install_libraries_with_deps(full_install_libs)

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

# Clean up
clean_triton_cache()

print("\n\033[92mInstallation completed successfully!\033[0m")
print("\033[92mYou can now run your KeyBERT GUI application with: python keybert_pyside.py\033[0m")

end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"\033[92m\nTotal installation time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\033[0m")