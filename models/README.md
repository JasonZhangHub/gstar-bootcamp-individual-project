# Models Directory

This directory stores trained model checkpoints and serialized model objects.

## Current Models

### ensemble_model_20250924_145122.pkl
- **Type**: Ensemble classifier (Random Forest + Gradient Boosting + SVM)
- **Created**: September 24, 2025
- **Size**: ~47 MB
- **Features**: Combined stylometric and semantic features
- **Performance**: [Add metrics after evaluation]

### quick_test_model.pkl
- **Type**: Quick test model for development
- **Size**: ~340 KB
- **Purpose**: Rapid testing and validation during development

## Model File Naming Convention

```
{model_type}_{YYYYMMDD}_{HHMMSS}.pkl
```

Examples:
- `ensemble_model_20250924_145122.pkl`
- `multitask_model_20250925_093045.pt`
- `neural_network_20250926_141530.pkl`

## Loading Models

```python
import pickle

# Load ensemble model
with open('models/ensemble_model_20250924_145122.pkl', 'rb') as f:
    model = pickle.load(f)

# For PyTorch models
import torch
model = torch.load('models/multitask_model.pt')
```

## Model Storage Guidelines

1. **Only store trained models** - No source code or scripts
2. **Use descriptive names** - Include model type and timestamp
3. **Document performance** - Add metrics to this README
4. **Version control** - Large files should use Git LFS
5. **Clean up old models** - Archive outdated checkpoints

## Performance Metrics

### Ensemble Model (2025-09-24)
- Accuracy: [To be added]
- F1-Score: [To be added]
- ROC-AUC: [To be added]
- Cross-validation: [To be added]

## Notes

- Source code for model training is in `src/model_training/`
- Test scripts are in `tests/`
- Feature extraction scripts are in `src/feature_engineering/`