# Fisher Information Matrix (FIM) Based Data Deletion

This project implements approximate data deletion using Fisher Information Matrix pruning to enable neural networks to "forget" specific training data points. The implementation focuses on privacy-preserving machine learning through selective parameter pruning based on data-specific importance scores.

## Theoretical Background

### Fisher Information Matrix (FIM)

The Fisher Information Matrix quantifies the amount of information that observable random variables carry about unknown parameters. In the context of neural networks, FIM measures the sensitivity of model parameters to specific data points, making it an ideal tool for identifying which weights are most important for remembering particular training examples.

**Mathematical Foundation:**
- For a parameter θ, the Fisher Information is: `F(θ) = E[∇log p(x|θ) ∇log p(x|θ)ᵀ]`
- In practice, we use the diagonal approximation: `F_ii ≈ (∂L/∂θᵢ)²`
- Higher FIM values indicate greater parameter importance for the given data

### Data Deletion via Pruning

The core methodology combines FIM computation with strategic weight pruning:

1. **FIM Computation**: Calculate Fisher Information scores for each parameter with respect to the deletion set
2. **Importance Ranking**: Rank parameters by their FIM scores (higher = more important for remembering target data)
3. **Selective Pruning**: Remove (zero out) the highest-scoring parameters to reduce the model's ability to memorize specific data points
4. **Fine-tuning**: Retrain the pruned model while maintaining the pruning mask to preserve the "forgetting" effect

### Privacy Evaluation

The effectiveness of data deletion is measured using:
- **Membership Inference Attacks (MIA)**: Evaluate whether an adversary can determine if specific data was used in training
- **Utility-Privacy Trade-offs**: Balance between model performance retention and privacy improvement
- **Comparative Analysis**: Compare against gold-standard retraining (training without the target data from scratch)

## Implementation Details

### Model Architecture
- Multi-Layer Perceptron (MLP) classifier
- Architecture: Input → Dense(128) → Dropout(0.2) → Dense(64) → Dropout(0.2) → Sigmoid
- Binary classification on Adult Income Dataset (>50K salary prediction)
- PyTorch implementation with Adam optimizer

### Pruning Pipeline
1. **Baseline Training**: Train model on complete dataset
2. **Deletion Set Selection**: Choose specific data points to "forget"
3. **FIM Analysis**: Compute diagonal Fisher Information for deletion set
4. **Global Thresholding**: Apply percentile-based pruning (10%, 20%, 30%, 40%)
5. **Mask Enforcement**: Maintain zero weights during fine-tuning
6. **Privacy Assessment**: Run membership inference attacks

### Key Features
- **Diagonal FIM Approximation**: Efficient computation using squared gradients
- **Global Pruning Strategy**: Threshold selection across all parameters
- **Mask Persistence**: Pruning masks maintained throughout fine-tuning
- **Comprehensive Evaluation**: Performance, privacy, and computational efficiency metrics
- **Automated Optimization**: Combined utility-privacy scoring for optimal pruning selection

## Results Summary

The implementation achieves:
- **Optimal Pruning Fraction**: 20% parameter removal
- **Utility Retention**: 100.2% (84.99% vs 84.83% baseline accuracy)
- **Privacy Improvement**: 2.9% reduction in MIA success rate
- **Combined Score**: 0.8473 (70% utility + 30% privacy weighting)
- **Computational Efficiency**: Minimal overhead for FIM computation and pruning

## Dataset

The project uses the **Adult Income Dataset** (Census Income):
- **Source**: UCI Machine Learning Repository
- **Task**: Binary classification (income >50K vs ≤50K)
- **Features**: Age, education, occupation, relationship, race, gender, capital gains/losses, hours per week
- **Size**: ~32,561 samples with 14 features
- **Preprocessing**: Label encoding for categorical variables, standard scaling for numerical features

## Installation and Usage

### Requirements Installation

```bash
pip install -r requirements.txt
```

### Running the Experiment

1. **Setup**: Ensure `adults.csv` is in the project directory
2. **Execution**: Open `problem22_fim_pruning.ipynb` in Jupyter Notebook or VS Code
3. **Run All Cells**: Execute all cells sequentially to run the complete experiment
4. **Results**: Results are automatically saved to `problem22_results.json` and `inference_report_summary.json`

### Expected Outputs

- **Model Files**: 
  - `model_full.pth` - Original trained model
  - `model_retrain.pth` - Gold standard (trained without deletion set)
  - `model_pruned_0.1.pth`, `model_pruned_0.2.pth`, etc. - Pruned models at different fractions
- **Results Files**:
  - `problem22_results.json` - Complete experimental results
  - `inference_report_summary.json` - Formatted analysis report
- **Visualizations**: Trade-off analysis plots showing utility vs privacy across pruning fractions

### Notebook Structure

The notebook is organized into the following sections:

1. **Cells 1-8**: Data loading, preprocessing, and initial model training
2. **Cells 9-15**: Fisher Information Matrix computation and analysis
3. **Cells 16-22**: Pruning experiments across multiple fractions
4. **Cells 23-27**: Membership inference attacks and evaluation
5. **Cells 28-41**: Comprehensive inference report generation with visualizations

## Key Contributions

1. **Practical FIM Implementation**: Efficient diagonal approximation suitable for production use
2. **Privacy-Utility Optimization**: Automated selection of optimal pruning parameters
3. **Comprehensive Evaluation**: Multi-faceted assessment including MIA resistance
4. **Reproducible Pipeline**: Complete experimental framework with detailed analysis
5. **Professional Reporting**: IEEE-style inference report with quantitative results

## Technical Specifications

- **Framework**: PyTorch 1.9+
- **Python**: 3.7+
- **Dependencies**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Computation**: CPU/GPU compatible (automatic device detection)
- **Memory**: ~2GB RAM recommended for full dataset processing
- **Runtime**: ~10-15 minutes for complete experiment on modern hardware

## Research Applications

This implementation supports research in:
- **Privacy-Preserving Machine Learning**: GDPR "right to be forgotten" compliance
- **Selective Unlearning**: Targeted removal of specific data points or classes
- **Model Compression**: FIM-guided pruning for efficient model deployment
- **Robustness Analysis**: Understanding parameter importance for model interpretability
- **Federated Learning**: Privacy-preserving updates in distributed training scenarios

All detailed results, analysis, and visualizations are contained within the notebook. Simply run all cells to reproduce the complete experimental pipeline and generate comprehensive reports.