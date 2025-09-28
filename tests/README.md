# Tests Directory

Unit tests and integration tests for the AI-Generated Text Detection project.

## Test Files

### Generation Tests
- **test_gpt2_generation.py** - Tests GPT-2 text generation pipeline
- **test_huggingface_generation.py** - Tests HuggingFace Inference API integration

### Model Tests
- **test_multitask_ensemble.py** - Tests multi-task ensemble classifier with synthetic data
- **test_quick_model.py** - Quick validation tests for model functionality

## Running Tests

### Run All Tests
```bash
python -m pytest tests/
```

### Run Specific Test
```bash
python tests/test_multitask_ensemble.py
python tests/test_quick_model.py
```

### Run with Coverage
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Test Categories

### Unit Tests
- Individual component testing
- Feature extraction functions
- Model predictions
- Data preprocessing

### Integration Tests
- End-to-end pipeline testing
- API integration
- Model loading and inference
- Feature combination

### Performance Tests
- Model inference speed
- Feature extraction efficiency
- Memory usage

## Adding New Tests

1. Create test file with `test_` prefix
2. Import necessary modules from `src/`
3. Write test functions with `test_` prefix
4. Use assertions to validate behavior

Example:
```python
import sys
sys.path.append('../src/model_training')
from multitask_ensemble import MultiTaskEnsembleClassifier

def test_model_initialization():
    model = MultiTaskEnsembleClassifier(n_features=100)
    assert model is not None
    assert model.n_features == 100
```

## Test Data

- Synthetic data generation for model testing
- Sample articles for feature extraction testing
- Mock API responses for integration testing

## CI/CD Integration

Tests should pass before merging:
```yaml
- name: Run tests
  run: python -m pytest tests/ -v
```