# Vision_System

An AI-based computer vision system for automated pick and place (INA700 project).

A camera above the table spots shapes on the paper, a Raspberry Pi works out **what** each shape is and **where** it is, and an ABB robot picks it up and drops it into its matching slot on the designated board.

![Cell setup: camera, robot, Raspberry Pi and designated positions](docs/setup.webp)

---

## 🔑 Key network info (read this first!)

| Device | IP address | Used for |
|---|---|---|
| **Raspberry Pi (Ethernet, robot network)** | **`192.168.125.207`** | ⭐ The robot talks to the Pi here (port **5000**). **This is what makes the robot move.** |
| Raspberry Pi (LAN / Wi-Fi) | `192.168.140.113` | SSH, copying files (`scp`) and the live camera feed |
| ABB robot controller | `192.168.125.1` | RobotStudio / FlexPendant connection |

- Live camera feed (while the vision script runs): **http://192.168.140.113:8080**
- The vision script listens on the Pi's robot-network address (`HOST = "192.168.125.207"` in `main_vision_yolo_without_ultralynics.py`), so the Pi must own that address on `eth0`. The RAPID side connects to the same address:
  - RAPID `ServerComms` module: `SocketConnect robotClient, "192.168.125.207", 5000;`

---

## 🔌 The Ethernet cable swap (important!)

There's **one Ethernet cable** going to the robot controller, and it's used for two different jobs:

| You want to… | Plug the robot's Ethernet cable into… |
|---|---|
| ✏️ **Edit the RAPID code** in RobotStudio | 🖥️ the **Windows desktop** |
| ▶️ **Run the robot** with vision | 🍓 the **Raspberry Pi** |

> ⚠️ **Always move the cable back to the Raspberry Pi before running the robot.** If the cable is still in the desktop, the robot can't reach the Pi at `.207` and the program will stall at "Requesting...".

---

## 🗂️ What's in this project

The cell runs from **two folders**: [`robot/`](robot) (the ABB controller side) and [`train_with_YOLO/`](train_with_YOLO) (the Raspberry Pi vision side and the model training). The other folders are older experiments and helper tools, kept for reference.

```
Vision_System/
├── robot/                  ← ⭐ used: RAPID code for the ABB controller
├── train_with_YOLO/        ← ⭐ used: vision script, YOLO model and training notebook
│   └── images/             training images and the Roboflow-labelled dataset
├── camera_setup/           helper scripts: check the camera view, collect training images
├── train_with_XGBoost/     old experiment (contour features + XGBoost)
├── train_with_random_forest/ old experiment (contour features + Random Forest)
└── docs/                   images for this README
```

### 🤖 `robot/` (runs on the ABB controller)

| File | What it does |
|---|---|
| [`robot/MainModule.mod`](robot/MainModule.mod) | **The robot program.** Asks the Pi which colours and shapes it sees, lets the operator choose on the FlexPendant, then picks that object and places it in the matching slot. |
| [`robot/ServerCommuncatio.txt`](robot/ServerCommuncatio.txt) | The `ServerCommuncatio` RAPID module (system module): socket helpers the main module uses to talk to the Pi (`RobotAsClientConnect`, `RobotClienSendMessage`, `RobotClientReciveMessage`, …). |

### 🍓 `train_with_YOLO/` (runs on the Raspberry Pi + training)

| File | What it does |
|---|---|
| [`main_vision_yolo_without_ultralynics.py`](train_with_YOLO/main_vision_yolo_without_ultralynics.py) | **The vision main script (the one we run on the Pi).** Grabs frames from the Basler camera, finds the ArUco marker, detects shapes with YOLO through OpenCV DNN, rejects anything outside the workspace, answers the robot on port **5000** and serves the live feed on port **8080**. |
| [`vision_common_with_yolo.py`](train_with_YOLO/vision_common_with_yolo.py) | Shared helpers imported by the main script: ArUco marker detection (`DICT_5X5_100`, **100 mm** marker), pixel→mm scale, pixel→robot offset, workspace utilities. |
| [`best.onnx`](train_with_YOLO/best.onnx) | The trained YOLOv8 model (classes: `circle`, `square`, `star`). Loaded with OpenCV DNN, so **no Ultralytics or PyTorch needed on the Pi**. |
| [`YOLO_Train.ipynb`](train_with_YOLO/YOLO_Train.ipynb) | Kaggle notebook: prepares the labelled dataset, trains the YOLOv8 model and exports it to `best.onnx`. |
| `data.yaml`, `args.yaml` | Dataset and training settings used by the notebook. |
| `results.csv`, `Box*_curve.png`, `confusion_matrix*.png`, `labels.jpg` | Training results and plots (useful for the report). |
| `yolov8n.pt`, `yolo26n.pt` | Pretrained YOLO weights the training starts from. Not needed on the Pi. |
| `main_vision_yolo_with_ultralynics.py` | Alternative vision script that uses the `ultralytics` package. Not used on the Pi. |
| `vision_common_with_yolo.py.txt` | Plain-text copy of `vision_common_with_yolo.py`. |
| [`images/`](train_with_YOLO/images) | The training images and the labels made in **Roboflow** (see *Training images & labelling* below). Not needed on the Pi. |

### 🧰 Other folders (reference / helpers)

| File | What it does |
|---|---|
| [`camera_setup/camera_setup.py`](camera_setup/camera_setup.py) | Starts the Basler camera and shows the live view in the browser. Use it to check the camera position. |
| [`camera_setup/capture_final_images.py`](camera_setup/capture_final_images.py) | Collects training images from the Basler camera (saves a frame every 10 s into `dataset/`, with a live preview on port 5000). |
| `train_with_XGBoost/`, `train_with_random_forest/` | Earlier experiments (classic contour features + XGBoost / Random Forest), including the old `main_vision.py`, `vision_common.py` and `shape_classifier.pkl`. Not used by the current YOLO pipeline. |

---

## 🖼️ Training images & labelling (Roboflow)

The images used to train the YOLO model are in [`train_with_YOLO/images/`](train_with_YOLO/images). They were taken with the Basler camera in the cell (using [`camera_setup/capture_final_images.py`](camera_setup/capture_final_images.py)), and the **labelling was done with [Roboflow](https://roboflow.com)**.

| Folder | What's in it |
|---|---|
| [`images/cleaned_labelled_dataset/`](train_with_YOLO/images/cleaned_labelled_dataset) | All **642** cleaned training images (circles, squares and stars at different positions and heights, alone or together, some with the ArUco marker). The file name says which shapes are in the picture, e.g. `capture__star_pos_3_2.jpg` or `square_star_circle_10.png`. |
| [`images/to_label/`](train_with_YOLO/images/to_label) | The **63** images the notebook couldn't label on its own (mostly pictures with several shapes), which were sent to Roboflow. |
| [`images/test.v1i.yolov8/`](train_with_YOLO/images/test.v1i.yolov8) | Those images **labelled in Roboflow** and exported in YOLOv8 format: `train/images/` + `train/labels/` (one `.txt` per image, one box per shape), plus `data.yaml`. Classes: `circle`, `square`, `star`. Roboflow project: [mary-wagura/test-8d5er](https://universe.roboflow.com/mary-wagura/test-8d5er) (CC BY 4.0). |

**How the dataset was built:**
1. 📸 Capture images of the shapes on the paper with the Basler camera.
2. 🏷️ In [`YOLO_Train.ipynb`](train_with_YOLO/YOLO_Train.ipynb), most images are labelled automatically: each shape's box comes from its contour and its class from the file name.
3. 🖍️ Images the notebook couldn't label with confidence were copied to `to_label/`, uploaded to **Roboflow**, labelled there by drawing a box around each shape, and exported as YOLOv8 (`test.v1i.yolov8/`). Roboflow resized them to 512×512 and added no augmentation.
4. 🧠 Both sets are combined, split 80/20 into train/val, and used to train YOLOv8, which is exported to `best.onnx`.

> To label new images: upload them to the Roboflow project, draw a box around each shape with the class `circle`, `square` or `star`, then **Export → YOLOv8** and add the export to the notebook's input data.

---

## 📥 Files you need to run the project

You **don't need the whole repo** to run the cell. Download/copy only these files:

| File (in this repo) | Copy it to… | Notes |
|---|---|---|
| [`train_with_YOLO/main_vision_yolo_without_ultralynics.py`](train_with_YOLO/main_vision_yolo_without_ultralynics.py) | 🍓 Raspberry Pi, e.g. `~/Downloads/` | The script you start. |
| [`train_with_YOLO/vision_common_with_yolo.py`](train_with_YOLO/vision_common_with_yolo.py) | 🍓 Raspberry Pi, **same folder** | Imported by the main script; keep this exact file name. |
| [`train_with_YOLO/best.onnx`](train_with_YOLO/best.onnx) | 🍓 Raspberry Pi, **same folder** | The model; the script loads it as `best.onnx`, so don't rename it. |
| [`robot/MainModule.mod`](robot/MainModule.mod) | 🤖 ABB controller (load in RobotStudio) | The main robot program. |
| [`robot/ServerCommuncatio.txt`](robot/ServerCommuncatio.txt) | 🤖 ABB controller (load in RobotStudio) | Socket module needed by `MainModule`. |

**On the Pi**, the three files must sit together in one folder:

```
~/Downloads/
├── main_vision_yolo_without_ultralynics.py
├── vision_common_with_yolo.py
└── best.onnx
```

Copy them from Windows with one command (PowerShell, after downloading them into your `Downloads` folder):

```powershell
scp "$env:USERPROFILE\Downloads\main_vision_yolo_without_ultralynics.py" "$env:USERPROFILE\Downloads\vision_common_with_yolo.py" "$env:USERPROFILE\Downloads\best.onnx" pi@192.168.140.113:~/Downloads/
```

Then check they arrived:

```bash
ls -l ~/Downloads/main_vision_yolo_without_ultralynics.py ~/Downloads/vision_common_with_yolo.py ~/Downloads/best.onnx
```

**On the robot controller**, open the station in RobotStudio (cable in the Windows desktop, see [the cable swap](#-the-ethernet-cable-swap-important)), load `MainModule.mod` and `ServerCommuncatio.txt` into the controller's task, apply the changes, then move the cable back to the Pi.

---

## 🛠️ Installation

### Raspberry Pi

```bash
# 1. Create a virtual environment (newer Raspberry Pi OS requires this for pip)
python3 -m venv ~/vision-env
source ~/vision-env/bin/activate

# 2. Install the Python packages
pip install opencv-contrib-python numpy flask pypylon
```

- `opencv-contrib-python` provides OpenCV, ArUco detection and the DNN module that runs the ONNX model.
- `pypylon` is the Basler camera driver.
- **No** `ultralytics` or PyTorch on the Pi. That's why we use the "without ultralytics" script.

Each time you open a new terminal on the Pi, activate the environment first:
```bash
source ~/vision-env/bin/activate
```

### 🐧 Raspberry Pi / Linux command cheat sheet

These are the commands used most often when setting up and running the vision system.

| Command | What it means | Example |
|---|---|---|
| `cd` | **Change directory.** Enters a folder so commands run from that location. | `cd ~/Downloads` |
| `cd ..` | Moves up one level to the parent directory. | `cd ..` |
| `cd ~` | Returns to your home directory. | `cd ~` |
| `ls` | **List.** Shows the files and folders in the current directory. | `ls` |
| `ls -l` | Shows files in a detailed list, including permissions, owner and file size. | `ls -l` |
| `ls -a` | Shows all files, including hidden files whose names start with `.`. | `ls -a` |
| `pwd` | **Print working directory.** Shows the full path of the directory you are currently in. | `pwd` |
| `sudo` | Runs a command with administrator privileges. For this project, `sudo apt update` refreshes the Raspberry Pi's package list before software is installed or updated. Use `sudo` carefully because it can change system-wide settings. | `sudo apt update`<br>`sudo apt install python3-venv` |
| `mkdir` | Creates a new directory (folder). | `mkdir vision-project` |
| `cp` | Copies a file or folder. | `cp model.onnx ~/Downloads/` |
| `mv` | Moves or renames a file or folder. | `mv best.onnx best_old.onnx` |
| `rm` | Removes a file. Be careful because deleted files may not be recoverable. | `rm old_log.csv` |
| `nano` | Opens a simple terminal text editor. Press `Ctrl+O` to save and `Ctrl+X` to exit. | `nano settings.py` |
| `source` | Activates a script or virtual environment in the current terminal. | `source ~/vision-env/bin/activate` |
| `python3` | Runs Python 3 programs. | `python3 main_vision_yolo_without_ultralynics.py` |
| `pip` | Installs Python packages. | `pip install flask` |
| `ssh` | Connects to the Raspberry Pi remotely from another computer. | `ssh pi@192.168.140.113` |
| `scp` | Copies files between computers over SSH. | `scp model.onnx pi@192.168.140.113:~/Downloads/` |
| `ping` | Tests whether another device can be reached over the network. | `ping 192.168.125.1` |
| `ip addr` | Displays the Pi's network interfaces and IP addresses. | `ip addr show eth0` |
| `ss` | Shows network connections and listening ports. Useful for checking ports `5000` and `8080`. | `ss -tlnp` |
| `clear` | Clears the terminal window. | `clear` |
| `history` | Shows previously entered commands. | `history` |

#### Useful examples for this project

```bash
# See where you are
pwd

# Go to the folder containing the vision files
cd ~/Downloads

# Check that the required files are present
ls -l

# Activate the Python virtual environment
source ~/vision-env/bin/activate

# Start the vision program
python3 main_vision_yolo_without_ultralynics.py

# Stop a running program
Ctrl + C
```

> **Important:** `sudo` gives a command administrator-level access. Use it only when necessary, and check the command carefully before pressing Enter. Commands such as `sudo rm` can permanently delete files or damage the system.

### Windows desktop
- **RobotStudio**, to edit and load the RAPID code on the controller.
- **OpenSSH client**, built into Windows 10/11. Check by typing `ssh` in PowerShell.

### Training (Kaggle)
- Run the notebooks on Kaggle with a **GPU** accelerator. The notebook installs `ultralytics` itself (`!pip install -q ultralytics`).

---

## 🍓 MainVision – Raspberry Pi vision server for the ABB robot

Runs on the Raspberry Pi (`pi-group12`). A Basler camera finds circles, squares and stars, and the script sends their
robot coordinates to the ABB controller over TCP. A live video feed is served over the network.

| What | Address |
|---|---|
| Pi on Wi-Fi (live feed) | `http://192.168.140.112:8080` |
| Pi on robot network (`eth0`) | `192.168.125.211` |
| TCP server for RAPID | `192.168.125.211:5000` |

### 1. Install dependencies (once)

```bash
sudo apt update
pip install opencv-contrib-python pypylon flask --break-system-packages
```

Check that everything installed:

```bash
python3 -c "import cv2, flask; from pypylon import pylon; print('OK', cv2.__version__, hasattr(cv2.aruco, 'ArucoDetector'))"
```

It should print `OK 4.x.x True` with OpenCV **4.7 or newer**.

> Use the pip package `opencv-contrib-python`, not `sudo apt install python3-opencv`: on Raspberry Pi OS the apt
> version is OpenCV 4.6, which doesn't have `cv2.aruco.ArucoDetector`, so the marker detection fails.
> The pip package is `pypylon` (not `pyplon`).

### 2. Connect the camera

The camera must be plugged into the **Raspberry Pi**, not the PC. The PC is only used to watch the live feed.

- **USB camera** (model ends in `um`/`uc`/`ubm`/`ubc`): plug it straight into a **blue USB 3 port** with a proper USB 3
  cable. Don't use a hub or extension.
- **GigE camera** (model ends in `gm`/`gc`/`gbm`/`gbc`): it needs its own power (PoE injector or power cable), because
  the Pi's Ethernet port supplies none. It must be cabled to the Pi, not to the lab network.

Check that the Pi sees the camera:

```bash
lsusb | grep 2676                      # USB: a line with 2676 = Basler camera found
python3 -c "from pypylon import pylon; print([d.GetModelName() for d in pylon.TlFactory.GetInstance().EnumerateDevices()])"
```

If the list is empty, check the cable, port and power, and watch `sudo dmesg -w` while you replug the camera.

### 3. Connect the Pi to the robot network

The script listens on `192.168.125.211`, which is the address the RAPID code connects to. The Pi must own this address
on `eth0`, or the TCP server fails with `OSError: [Errno 99] Cannot assign requested address`.

1. Connect an Ethernet cable from the Pi's `eth0` to the robot controller's network.
2. Give `eth0` the robot-network address:

   ```bash
   sudo ip addr add 192.168.125.211/24 dev eth0
   ip addr show eth0        # should show: state UP  and  inet 192.168.125.211/24
   ```

3. Check you can reach the controller (replace with your controller's IP if it is different):

   ```bash
   ping 192.168.125.1
   ```

The `ip addr add` command is lost on reboot. To make the address permanent:

```bash
nmcli con show                                   # find the wired connection name
sudo nmcli con mod "Wired connection 1" ipv4.method manual ipv4.addresses 192.168.125.211/24
sudo nmcli con up "Wired connection 1"
```

> **GigE camera + robot:** both can't each have their own cable in the single `eth0` port. Use a switch or a
> USB-Ethernet adapter for the second connection. With a USB camera, `eth0` is free for the robot.

### 4. Run

These files must be in the same folder: `main_vision_yolo_without_ultralynics.py`, `vision_common_with_yolo.py` and `best.onnx` (see [Files you need to run the project](#-files-you-need-to-run-the-project)).

```bash
cd ~/Downloads
python3 main_vision_yolo_without_ultralynics.py
```

Expected startup output:

```
Model warm-up: ... ms (not counted)
TCP Server listening on 192.168.125.211:5000
Live Video Feed available at: http://192.168.140.112:8080
Camera active. Waiting for robot...
```

Open the live feed from a PC on the same Wi-Fi: `http://192.168.140.112:8080`. Stop the script with `Ctrl + C`.

To test without the robot, set `HOST = "0.0.0.0"` in the script so it listens on every interface.

### 5. Troubleshooting

| Error / symptom | Meaning | Fix |
|---|---|---|
| `python : The term 'python' is not recognized` (Windows) | Python is not on PATH | Use `py`, or add `...\AppData\Local\Programs\Python\Python3xx` and its `Scripts` folder to PATH |
| `No matching distribution found for OpenCV` | Wrong package name | `pip install opencv-contrib-python --break-system-packages` |
| `AttributeError: ... 'ArucoDetector'` | OpenCV is older than 4.7 (e.g. the apt `python3-opencv`) | `sudo apt remove python3-opencv`, then `pip install opencv-contrib-python --break-system-packages` |
| `No device is available ... (file 'TlFactory.cpp', line 694)` | pylon found **no camera** (the file name is from Basler's internal C++ SDK) | See step 2: cable, USB 3 port, power, camera plugged into the Pi |
| `OSError: [Errno 99] Cannot assign requested address` | The Pi doesn't have the IP in `HOST` | See step 3: give `eth0` the address `192.168.125.211` |
| `OSError: [Errno 98] Address already in use` | Another program is already using the port | `sudo ss -tlnp \| grep 5000` to find it, then stop it (or wait ~30 s after a restart) |
| `eth0` shows `NO-CARRIER` | No live cable link | Check the cable, and the camera/switch power |
| Live feed doesn't load | Server not running or not reachable | On the Pi: `ss -tlnp \| grep 8080`. On Windows: `Test-NetConnection 192.168.140.112 -Port 8080` |

---

## 📤 Copying files from Windows to the Pi (SSH / SCP)

Run these in **PowerShell** on the Windows desktop. Replace `pi` with the Pi's username if it's different.

**Copy one file** from your Windows `Downloads` to the Pi's `Downloads`:
```powershell
scp "$env:USERPROFILE\Downloads\best.onnx" student@192.168.140.113:~/Downloads/
```

**Copy several files at once:**
```powershell
scp "$env:USERPROFILE\Downloads\main_vision_yolo_without_ultralynics.py" "$env:USERPROFILE\Downloads\vision_common_with_yolo.py" pi@192.168.140.113:~/Downloads/
```

**Copy a whole folder:**
```powershell
scp -r "$env:USERPROFILE\Downloads\my_folder" pi@192.168.140.113:~/Downloads/
```

**Log in to the Pi:**
```powershell
ssh pi@192.168.140.113
```

> 💡 The first time, it asks *"Are you sure you want to continue connecting?"*: type `yes`, then enter the Pi's password.

---

## 📷 Checking the camera view

To just see what the camera sees (e.g. after moving the camera, paper or marker), run [`camera_setup/camera_setup.py`](camera_setup/camera_setup.py) on the Pi and open **http://192.168.140.113:8080**:

```bash
python3 camera_setup.py
```

Stop it with `Ctrl + C` before starting the vision script, because both use the camera and port 8080.

---

## ▶️ Running the cell, step by step

1. 🔌 **Cable:** robot Ethernet cable plugged into the **Raspberry Pi**.
2. 🍓 **Start the vision script on the Pi:**
   ```bash
   ssh pi@192.168.140.113
   source ~/vision-env/bin/activate
   cd ~/Downloads
   python3 main_vision_yolo_without_ultralynics.py
   ```
   Wait for `TCP Server listening on 192.168.125.207:5000` and `Camera active`.
3. 👀 **Check the live feed** at http://192.168.140.113:8080. You should see the red **Origin** dot on the marker and green dots on the shapes.
4. 🤖 **Start the robot program** (`MainModule`) on the FlexPendant. It connects to the Pi ("I LIVE").
5. 🎨 **Choose a colour, then a shape** on the FlexPendant (see the next section). The menus only list what the camera sees.
6. 🟡 **Press the yellow button** when the pendant says *"Press yellow button to pick the …"*.
7. 🔁 The robot places the object and goes straight back to the colour menu. No restart needed.
8. 🛑 To stop: stop the program on the FlexPendant, then press `Ctrl + C` in the Pi terminal.

---

## 🎨 How a picking cycle works

Each cycle the operator chooses **one object** by colour and shape. The menus are built from what the camera detects, so you can only choose something that's actually on the paper.

```
Which colour should the robot pick?        Which red shape should the robot pick?
1 = blue                                   1 = square
2 = red                                    2 = star
0 = Refresh the list                       0 = Refresh the list
9 = Board emptied (reset slot counters)    Enter a number:  2
Enter a number:  2
```

| Robot sends | Pi answers |
|---|---|
| `LIST_COLORS` | colours it sees, e.g. `blue,red` (or `NONE`) |
| `LIST_SHAPES,red` | red shapes it sees, e.g. `square,star` (or `NONE`) |
| `REQUEST_COORDS,red,star` | `star,x,y,angle` (or `NO_TARGET,0,0,0`) |

- **Colours:** black, red, green, blue, white. **Shapes:** circle (→ `p20`), star (→ `p30`), square (→ `p40`).
- **Two identical objects** (e.g. two red squares): the one closest to the marker is picked first; choose it again next cycle for the other. Extra shapes in the same slot are released 10 mm higher each time.
- **After emptying the board**, choose **9** in the colour menu to reset the slot counters.
- **0** refreshes the menu, e.g. after putting new shapes on the paper.

### ⏱️ Inference time (must be ≤ 3 s)

The vision script prints a line for every frame, e.g.:
```
[INFERENCE] YOLO model: 412 ms | total detection: 655 ms | 2 shape(s) | OK (limit 3 s) | avg 640 ms, max 702 ms, over limit 0/57
```
- **Total detection** = camera frame in → shapes, colours and robot X/Y out. This is the number checked against the 3 s limit.
- Frames over the limit print a `[WARNING]` line, and the live feed shows the time in red.
- Every frame is also saved to `inference_log.csv` (next to the script) as evidence for the report.

---

## 📐 Calibration notes

These must stay consistent between the robot and the Python script.

| Setting | Value / rule |
|---|---|
| Tool | `tool1` |
| Work object | `wobj1`. **All** robot targets and readings use this. |
| FlexPendant jogging | Tool = `tool1`, Work object = `wobj1`, Coordinate system = **Work object** |
| Image ↔ robot axes | robot **+X = image down**, robot **+Y = image right** |
| Marker position | Taught RAPID target `aruco` (cup centred on the marker, touching the paper). Python sends positions **relative to the marker**, so nothing in Python needs changing. |
| Marker size | 100 mm (outer black square), `DICT_5X5_100` |
| Pick height | Taught RAPID target `shapeTop`: cup pressed onto a real shape until the vacuum holds it. Its Z is used directly, whether it's positive or negative. |
| Shape thickness | 10 mm (only used to stack extra shapes in a slot). |
| Shape names | Python sends lowercase: `circle`, `square`, `star` |

**If the paper, marker, tool or work object changes:** re-teach two points on the FlexPendant (`tool1` + `wobj1`, then **Modify Position**). No code changes, in Python or RAPID:
1. **`aruco`**: jog the cup onto the marker's centre, touching the paper.
2. **`shapeTop`**: put a shape on the paper, jog the cup down onto it, turn the valve on and keep going down until the vacuum holds the shape. Modify Position there.

**If the camera gets bumped or moved:** re-check the marker position and the axis directions before running.

---

## ⚠️ Safety rules (learned the hard way)

- **Check every taught target's Z** before running. A home point with a negative Z once drove the tool into the table and broke it.
- After **any** change (new tool, new work object, re-taught points), run first in **Manual, reduced speed**, stepping one instruction at a time with your hand on the enabling device.
- After replacing the tool, **re-calibrate the TCP** (`tool1`) and re-check `p20`–`p40` and the `aruco` target.
- Keep hands clear of the work area while the program runs. The yellow button confirms each pick.

---

## 🧯 Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Pendant stuck at after *"I LIVE"*, or no colour menu appears | Cable is still in the desktop, or the vision script isn't running. Plug the cable into the Pi and start the script. |
| Pendant keeps saying *"No pickable shapes on the paper"* | The camera sees no circle/square/star with a recognised colour. Check the live feed and the `colour=…` debug lines on the Pi. |
| A shape is on the paper but missing from the menu | Its colour came out `unknown`, or it failed the geometry check. Check the `colour=…` / `corners=…` debug lines and tune the thresholds. |
| Cup stops above the shape / vacuum doesn't hold | Re-teach `shapeTop` with the cup pressed onto a shape until the vacuum holds. If it still slips, set `EXTRA_SQUEEZE` to 0.5–1. |
| `no marker` printed on the Pi | Marker not in view, covered or badly lit. |
| Shapes detected but the robot goes to the wrong place | The paper, marker or camera moved. Re-teach `aruco` (cup on the marker centre). |
| Robot picks the right spot but places in the wrong slot | Re-teach `p20` (circle), `p30` (star), `p40` (square) in `wobj1`. |
| Every shape skipped as *"Unknown shape"* | Shape names don't match: RAPID must compare lowercase `circle` / `star` / `square`. |
| Triangle or hexagon gets picked | The geometry check in the vision script is off or needs tuning. Check the printed `corners=… circle_fill=…` values. |
| `scp` / `ssh` times out | Use the Pi's LAN IP `192.168.140.113`, not `.207`, and check the Pi is powered on. |

## Acknowledgements
*This Project was in collaboration with Elias & Kelvin.*
