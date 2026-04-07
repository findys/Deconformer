
### 🪟 For Windows Users (Recommended: WSL 2)
To run Deconformer on Windows, we highly recommend using **WSL 2 (Windows Subsystem for Linux)**. This allows you to run the original Linux bash scripts without any modification.

**Step 1: Install WSL**
Open PowerShell as Administrator and run:
```powershell
wsl --install
```
*(Restart your computer if prompted)*

**Step 2: Launch Ubuntu**
Search for "Ubuntu" in your Start Menu and open it. Set up your username and password.

**Step 3: Install Dependencies inside Ubuntu**
Inside the Ubuntu terminal, update packages and install Python/Pip:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip git -y
pip3 install torch scanpy
```

**Step 4: Clone and Run**
```bash
git clone https://github.com/findys/Deconformer.git
cd Deconformer
# Now you can run the original bash script directly!
python deconformer_inference.py --model adult_model --input example_input/PE2020.TPM.txt --output inference_results/test_output.txt
```
