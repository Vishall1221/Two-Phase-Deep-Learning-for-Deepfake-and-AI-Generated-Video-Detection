Final combined project UI.

Pipeline:
1. Phase 1: AI Generated vs Camera Captured
   - Uses the two-branch ResNet18 residual + FFT model.
2. Phase 2: Real vs Deepfake
   - Runs only when Phase 1 predicts camera-captured.
   - Uses the Xception + EfficientNet-B2 weighted ensemble.

Folders:
- models/
  Contains all copied local model files needed by this UI.
- uploads/
  Stores uploaded videos.
- static/previews/
  Stores generated preview images and probability charts.
- templates/
  Flask HTML template.

Run:
- Double-click run_final_project_ui.bat
or
- Open a terminal in this folder and run:
  C:\Users\jaira\AppData\Local\Programs\Python\Python311\python.exe app.py

Default Flask URL:
- http://127.0.0.1:5000
