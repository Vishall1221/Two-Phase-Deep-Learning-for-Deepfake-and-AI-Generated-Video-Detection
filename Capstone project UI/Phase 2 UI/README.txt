Standalone deepfake UI deployment directory.

This folder is self-contained for the final best ensemble UI.
It does not need to read checkpoint files from the old training-run directories.

Structure:
- deepfake_ui_app.py
  Streamlit frontend UI
- ensemble_video_predictor.py
  Local backend inference pipeline
- run_deepfake_ui.bat
  Windows launcher
- models/
  - xception_best_model.pth
  - efficientnet_b2_best_model.pth
  - ensemble_config.json
- assets/
  - final_best_ensemble_architecture.png
- runtime/
  Created automatically when the UI runs

How to start:
- Double-click run_deepfake_ui.bat
or
- Run:
  C:\Users\jaira\AppData\Local\Programs\Python\Python311\python.exe -m streamlit run deepfake_ui_app.py

Expected browser URL:
- http://127.0.0.1:8501

Final ensemble used here:
- Xception weight = 0.45
- EfficientNet-B2 weight = 0.55
- Best test accuracy = 96.67%
