# Phase 7: The Ultimate "Day Zero" Execution Spec Sheet

**Goal:** To properly utilize Supercomputer access to test, quantize, and simulate the 1.58-bit Bit-Serial Kernel. 
*Note: This document is written for someone who has never touched a Linux supercomputer or terminal before. Follow every single instruction exactly as written.*

---

## 0. Initial Setup (Before You Do Anything)

- [ ] **Step 1: Connect to the Internet**
  * **Action:** Open a web browser (like Google Chrome or Edge) on the lab computer and go to `google.com` to verify you have internet access.
  * **Common Mistake:** Forgetting to log into the college's Wi-Fi portal.
  * **Solution:** If the browser shows "No Internet," look for the college Wi-Fi login page, enter your student credentials, and test `google.com` again.

- [ ] **Step 2: Open the Terminal**
  * **Action:** Click the Windows "Start" menu at the bottom left (or hit the Windows key) and type `Terminal` or `Command Prompt`. Press Enter to open it.
  * **What this does:** Opens a black box where you type text commands to control the computer directly, bypassing graphical folders.

- [ ] **Step 3: SSH into the Supercomputer**
  * **Action:** In the black box, type `ssh your_username@supercomputer_address` and press Enter. (You must ask your lab admin for the exact username and address).
  * **Action:** Type your password. *Note: As you type your password, absolutely nothing will show on the screen (no dots, no stars). This is normal Linux security. Just type it confidently and hit Enter.*
  * **Common Mistake:** "Connection Refused" or "Timeout" error.
  * **Solution:** Check if you misspelled the supercomputer address, or check with the IT desk if you need to be connected to a specific college VPN to access the supercomputer.

---

## 1. Morning Session: Environment & Data Download

- [ ] **Step 4: Create a "Virtual Bubble" for Python**
  * **Action:** Type `python3 -m venv ternary_env` and press Enter.
  * **What this does:** Creates an isolated folder called `ternary_env` so the software you install doesn't break the supercomputer's main system.
  * **Common Mistake:** Error saying `python3 is not recognized` or `Command not found`.
  * **Solution:** Try typing `python -m venv ternary_env` (dropping the '3'). If that fails, the supercomputer uses a module system. Type `module load python` and try Step 4 again.

- [ ] **Step 5: Enter the Bubble**
  * **Action:** Type `source ternary_env/bin/activate` and press Enter.
  * **What to look for:** Look at the far left of the text you are typing. Your prompt text MUST now start with `(ternary_env)`. If it doesn't, you aren't in the bubble. Do not proceed until you see it.

- [ ] **Step 6: Install AI Tools**
  * **Action:** Type `pip install torch transformers huggingface_hub numpy` and press Enter.
  * **What to do next:** Wait. Text will scroll by rapidly as it downloads packages from the internet. Wait until your prompt returns and stops moving.

- [ ] **Step 7: Download the QAT BitNet LLM**
  * **What this does:** Downloads the natively trained 1.58-bit model directly from HuggingFace. We are using this specific model: [herry90/llama-base-3b-bitnet](https://huggingface.co/herry90/llama-base-3b-bitnet/blob/main/model.safetensors).
  * **Action A:** Type `python3` and press Enter. The prompt will change to `>>>` (You are now inside the Python language).
  * **Action B:** Copy and paste the following exactly, then hit Enter twice to run it:
    ```python
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo_id="herry90/llama-base-3b-bitnet", filename="model.safetensors")
    print(model_path)
    exit()
    ```
  * **Common Mistake:** "No module named huggingface_hub" error.
  * **Solution:** You forgot to run Step 6 or you are not inside the `(ternary_env)` bubble. Ensure `(ternary_env)` is on your screen, run Step 6, and try again.

- [ ] **Step 8: Pack the Data for your Pendrive**
  * **Action:** Type `python3 pack_weights.py` and press Enter.
  * **What this does:** Runs our custom script to crush the downloaded massive file into pure binary bits.
  * **What to look for:** When it finishes, type `ls -lh` to list your files. Look for `packed_weights.bin`. It should be roughly 600MB to 1.5GB in size.

---

## 2. Mid-Day Session: Testing the Math

- [ ] **Step 9: Compile our C++ Code into Python**
  * **Action:** Type `python3 setup.py install` and press Enter.
  * **Common Mistake:** "gcc compiler not found", "error: command 'gcc' failed", or general red text.
  * **Solution:** The supercomputer requires you to load a C++ compiler. Type `module load gcc` and then try Step 9 again.

- [ ] **Step 10: Run the Mathematical Test**
  * **Action:** Type `python3 layer_test.py` and press Enter.
  * **What this does:** It tests the standard heavy PyTorch math against our custom C++ bit-serial logic.
  * **What to look for:** Wait for the script to finish. Look at the very last line of text it prints. It MUST say `True`. 
  * **Common Mistake:** It prints `False`.
  * **Solution:** Stop everything. If it prints `False`, our C++ math is mathematically incorrect and we cannot proceed to benchmarking. You must open `poc_ternary_mac.cpp` and debug the logic.

---

## 3. Afternoon Session: Rigorous Benchmarking

- [ ] **Step 11: Download the Hardware Simulator (gem5)**
  * **Action:** Type `git clone https://gem5.googlesource.com/public/gem5` and press Enter.
  * **What to do next:** When it finishes downloading, type `cd gem5` and hit Enter.

- [ ] **Step 12: Compile the Simulator**
  * **Action:** Type `scons build/ARM/gem5.opt -j$(nproc)` and press Enter.
  * **What to look for:** This will take 10-20 minutes. Go get a coffee. Do not touch the keyboard until the text stops scrolling and it says "Done building".
  * **Common Mistake:** "scons: command not found" error.
  * **Solution:** Type `pip install scons` or `module load scons` and try Step 12 again.

- [ ] **Step 13: Run the Baseline Edge Simulation**
  * **Action:** Type `gem5 run_baseline_sim.py` and press Enter.
  * **What to do next:** Wait. Hardware simulation is incredibly slow because it calculates every single electrical clock cycle.

- [ ] **Step 14: Run the Bit-Serial Edge Simulation**
  * **Action:** Type `gem5 run_optimized_sim.py` and press Enter. Wait for it to finish.

- [ ] **Step 15: Extract the Data**
  * **Action:** Type `python3 extract_metrics.py` and press Enter.
  * **What to look for:** Type `ls`. Ensure `results.csv` has appeared in your folder. This is the spreadsheet that proves your thesis.

---

## 4. Final Cleanup & Backup

- [ ] **Step 16: Plug in your Pendrive**
  * **Action:** Physically plug your 16GB pendrive into the lab computer. Wait for it to show up on the desktop or in the File Explorer.

- [ ] **Step 17: Secure the Files**
  * **Action:** If you used SSH through a terminal like MobaXterm, use the drag-and-drop file browser on the left side of the screen. Drag `packed_weights.bin`, `results.csv`, and all your `.cpp` and `.py` files onto your physical pendrive.

- [ ] **Step 18: Delete Massive Cache Files**
  * **Action:** In your black terminal box, type `rm ~/.cache/huggingface/hub/models--herry90--llama-base-3b-bitnet/ -rf` and press Enter.
  * **What this does:** Deletes the massive multi-gigabyte models from the supercomputer so you don't get banned for hoarding storage limits.

- [ ] **Step 19: Clean your Workspace**
  * **Action A:** Type `deactivate` and press Enter to exit your virtual bubble.
  * **Action B:** Type `rm -rf ternary_env/` and press Enter to delete the bubble.
  * **Action C:** Type `cd ..` then `rm -rf gem5/` and press Enter to delete the massive simulator folder.
  
- [ ] **Step 20: Log Out**
  * **Action:** Type `exit` and hit Enter. The black box will close. You are done!
