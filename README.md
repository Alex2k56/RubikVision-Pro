# 🧩 RubikVision Pro

A machine learning-based Rubik's Cube solver. RubikVision Pro combines **Computer Vision**, **Machine Learning** and **algorithmic problem-solving** to scan a scrambled Rubik's Cube with a webcam and guide you through solving it, step by step.

## 🔍 Features

- 📸 **Scan all six faces** of a Rubik's Cube using your webcam
- 🎨 **Color detection** with OpenCV and a **Random Forest classifier**
- ✏️ **Manual correction** of any misdetected sticker before solving
- ✅ **Cube validation** (sticker counts, distinct centers) before solving
- 🧠 **Solution generation** using Kociemba's two-phase algorithm
- 👣 **Step-by-step solving guidance** with a plain-language explanation for each move
- 🔁 **Self-improving model**: confirmed scans can be saved as training data

## 🛠️ Tech Stack

| Area | Technology |
|------|------------|
| Backend | Python, Django |
| Computer Vision | OpenCV, NumPy |
| Machine Learning | Scikit-learn (Random Forest), Pandas, Joblib |
| Solver | Kociemba's two-phase algorithm (`RubikTwoPhase`) |
| Frontend | HTML, CSS, JavaScript |

## ⚙️ How It Works

1. **Scan**: the browser captures a frame from the webcam, cropped to a square around the cube face.
2. **Extract features**: OpenCV splits the image into a 3x3 grid and takes the median LAB and HSV color of the center of each sticker.
3. **Classify**: a Random Forest predicts one of six colors (W, Y, R, O, G, B). Until a model is trained, simple HSV threshold rules are used as a fallback.
4. **Review**: the user checks the detected colors and can click any sticker to fix it.
5. **Validate**: the app checks that there are 6 distinct centers and exactly 9 stickers of each color.
6. **Solve**: the 54-sticker state is converted to the facelet string (U, R, F, D, L, B) and solved with Kociemba's two-phase algorithm.
7. **Guide**: the solution is displayed one move at a time with navigation buttons.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A webcam
- A modern browser (Chrome or Edge recommended)

### Installation

```bash
git clone https://github.com/Alex2k56/rubikvision-pro.git
cd rubikvision-pro

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
python manage.py migrate
```

### First run: solver tables

`RubikTwoPhase` generates its lookup tables the first time it runs. This can take a few minutes and some disk space. Prepare them once before using the app:

```bash
python -c "import twophase.solver as sv; print(sv.solve('DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD', 20, 2))"
```

### Run

```bash
python manage.py runserver
```

Open <http://127.0.0.1:8000> and allow camera access. The camera works only on `localhost` or over HTTPS.

## 📖 Usage

1. Hold the cube so that the requested center faces the camera and fits inside the white frame.
2. Press **Scan face**, then check the detected colors (click a square to change its color).
3. Press **Confirm face** and repeat for all six faces: Up, Right, Front, Down, Left, Back.
4. Press **Solve the cube** and follow the moves one by one.

Move notation: a letter is the face to turn, no sign means 90° clockwise, `'` means counter-clockwise, `2` means 180°.

## 🤖 Training the Color Classifier

The app works out of the box with HSV rules, but accuracy improves a lot if the model is trained on **your** cube and lighting.

1. Keep **"Save confirmed colors as training data"** checked while solving a few cubes. Every solve adds 54 labeled samples to `solver/dataset.csv`.
2. After 3–5 cubes, train the model:

```bash
   python train_model.py
```

3. Restart the server. The Random Forest model (`solver/color_model.joblib`) is now used automatically.

## 📁 Project Structure

```
rubikvision-pro/
├── manage.py
├── requirements.txt
├── train_model.py          # trains the Random Forest color classifier
├── rubikvision/            # Django project settings and root URLs
└── solver/
    ├── cv_utils.py
