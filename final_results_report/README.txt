Final report bundle for the best baseline and ensemble experiments.

Included folders:
- xception_baseline
- efficientnet_b2
- ensemble_xception_efficientnet_b2

What is copied here:
- accuracy and validation plots
- standalone validation-vs-epoch plots for the two trained single models
- confusion matrices
- classification reports
- summary CSV/JSON metric files
- extra useful report figures where available

Source run directories:
- Xception baseline:
  C:\Users\jaira\Desktop\Project\artifacts\merged_generalized_deepfake_1500\training_runs\xception_1776369649
- EfficientNet-B2:
  C:\Users\jaira\Desktop\Project\artifacts\merged_generalized_deepfake_1500\training_runs\efficientnet_b2_1776402608
- Best ensemble:
  C:\Users\jaira\Desktop\Project\artifacts\merged_generalized_deepfake_1500\training_runs\ensemble_xception_efficientnet_b2_1776418065

Important note:
- The ensemble is a post-training weighted soft-voting model, so it does not have an epoch-wise training curve.
- For the ensemble folder, ensemble_weight_search.png is included instead as the most relevant validation figure.
