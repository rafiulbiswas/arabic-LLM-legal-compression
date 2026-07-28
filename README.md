# arabic-LLM-legal-compression

````markdown
# Three-Stage Compression for Arabic Legal Language Models

This repository contains the code, experiment notebooks, configurations, and result files for a study on compressing large language models for Arabic legal understanding.

The study evaluates a systematic three-stage compression pipeline on the **ArabLegalEval** benchmark:

1. **Domain adaptation**
2. **Quantization**
3. **Knowledge distillation**

The pipeline is applied to three open-source language models:

- **FANAR-9B**
- **JAIS-8B**
- **Qwen2.5-7B**

All experiments are evaluated under both:

- **With-context evaluation**, where the reference legal passage is provided
- **No-context evaluation**, where the model must rely on retained legal knowledge

This distinction allows the study to separate reading-comprehension and extraction ability from parametric legal knowledge.

---

## Repository Structure

```text
.
├── results_QWEN_stage1/
├── results_fanar_stage1/
├── results_fanar_stage2/
├── results_fanar_stage3/
├── results_jais_stage1/
├── results_jais_stage2/
├── results_jais_stage3/
├── results_qwen_stage3/
│
├── stage1_QWEN.ipynb
├── stage1_arablegal_baseline.ipynb
├── stage1_arablegal_jais.ipynb
│
├── stage2_quantization-QWEN.ipynb
├── stage2_quantization-jais.ipynb
├── stage2_quantization_fanar.ipynb
│
├── stage3_QWEN_distillation.ipynb
├── stage3_QWEN_distillation-Copy1.ipynb
├── stage3_fanar_distillation.ipynb
├── stage3_jais_distillation.ipynb
├── stage3_jais_distillation_fixed.ipynb
│
├── .gitignore
└── README.md
````

---

## Compression Pipeline

### Stage 1: Domain Adaptation

For each base model, we evaluate the following adaptation strategies:

* Continual pre-training (**CPT**)
* Supervised fine-tuning (**SFT**)
* Sequential adaptation (**CPT → SFT**)

The strongest adapted checkpoint for each base model is selected as the teacher model for the subsequent compression stages.

Relevant notebooks:

```text
stage1_QWEN.ipynb
stage1_arablegal_baseline.ipynb
stage1_arablegal_jais.ipynb
```

Stage 1 result directories:

```text
results_QWEN_stage1/
results_fanar_stage1/
results_jais_stage1/
```

---

### Stage 2: Quantization

The selected teacher model from Stage 1 is evaluated under the following quantization settings:

* **INT8**, using `bitsandbytes` `LLM.int8()`
* **INT4 NF4**
* **INT4 FP4**

The objective is to measure the trade-off between memory efficiency and Arabic legal task performance.

Relevant notebooks:

```text
stage2_quantization-QWEN.ipynb
stage2_quantization-jais.ipynb
stage2_quantization_fanar.ipynb
```

Stage 2 result directories:

```text
results_fanar_stage2/
results_jais_stage2/
```

---

### Stage 3: Knowledge Distillation

The selected teacher model from Stage 1 is used to transfer knowledge into progressively smaller student models.

The distillation experiments investigate how much Arabic legal competence can be retained as model size is reduced.

Optional INT4 quantization may also be applied to student models after distillation.

Relevant notebooks:

```text
stage3_QWEN_distillation.ipynb
stage3_QWEN_distillation-Copy1.ipynb
stage3_fanar_distillation.ipynb
stage3_jais_distillation.ipynb
stage3_jais_distillation_fixed.ipynb
```

Stage 3 result directories:

```text
results_fanar_stage3/
results_jais_stage3/
results_qwen_stage3/
```

---

## Benchmark

The experiments use **ArabLegalEval**, a multitask Arabic legal evaluation benchmark constructed from Saudi legal and governmental sources, including documents from the Saudi Ministry of Justice and the Board of Experts.

The benchmark includes:

### Multiple-Choice Questions

Legal multiple-choice questions evaluated using:

```text
Accuracy
```

The MCQ experiments are conducted under both with-context and no-context protocols.

### NajizQA

Open-ended Arabic legal question answering evaluated using:

```text
ROUGE-L
```

### Arabic LegalBench Tasks

Arabic translations of LegalBench tasks are included in the broader ArabLegalEval benchmark but are not evaluated in the experiments included in this repository.

---

## Evaluation Protocols

### With-Context Protocol

The model receives the relevant legal source passage together with the question.

In this setting, the answer may often be extracted from the provided passage. Therefore, performance primarily reflects:

* Reading comprehension
* Information extraction
* Context utilization

It should not be interpreted solely as evidence of retained legal knowledge.

### No-Context Protocol

The source passage is removed, and the model receives only the question and answer options.

This setting is designed to isolate the model's retained, parametric knowledge of Saudi laws and regulations.

---

## Environment Setup

A Python environment with GPU support is recommended.

Example setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

Install the main dependencies:

```bash
pip install \
  torch \
  transformers \
  datasets \
  accelerate \
  bitsandbytes \
  peft \
  trl \
  evaluate \
  rouge-score \
  scikit-learn \
  pandas \
  numpy \
  matplotlib \
  sentencepiece \
  huggingface-hub \
  jupyter
```

The exact dependency versions used in each experiment can also be inspected from the corresponding notebooks.

---

## Hugging Face Authentication

Some models require access through Hugging Face.

Do not place access tokens directly inside notebooks. Set the token through an environment variable:

```bash
export HF_TOKEN="your_hugging_face_token"
```

The notebooks can then access it using:

```python
import os
from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise RuntimeError("HF_TOKEN is not set.")

login(token=hf_token)
```

---

## Running the Experiments

The experiments should be executed sequentially.

### 1. Run Stage 1

Run the adaptation notebook corresponding to the selected base model:

```text
stage1_QWEN.ipynb
stage1_arablegal_baseline.ipynb
stage1_arablegal_jais.ipynb
```

Select the strongest teacher checkpoint based on the ArabLegalEval validation or test results reported in the notebook.

### 2. Run Stage 2

Use the selected Stage 1 teacher checkpoint as input to the corresponding quantization notebook:

```text
stage2_quantization-QWEN.ipynb
stage2_quantization-jais.ipynb
stage2_quantization_fanar.ipynb
```

### 3. Run Stage 3

Use the selected Stage 1 teacher checkpoint for knowledge distillation:

```text
stage3_QWEN_distillation.ipynb
stage3_fanar_distillation.ipynb
stage3_jais_distillation_fixed.ipynb
```

The notebooks containing `Copy1` or older versions are retained for experiment traceability. The notebook with the clearest final or fixed designation should be used for reproduction.

---

## Results

The result directories contain experiment outputs such as:

* Evaluation metrics
* Model predictions
* Per-task results
* With-context results
* No-context results
* Quantization comparisons
* Student-model performance
* Training and evaluation summaries

The directories are organized by model family and compression stage.

```text
results_<model>_stage<stage>/
```

Where:

* `<model>` is `fanar`, `jais`, or `qwen`
* `<stage>` is `1`, `2`, or `3`

---

## Reproducibility Notes

Model outputs may vary slightly depending on:

* GPU architecture
* CUDA version
* PyTorch version
* `transformers` version
* Quantization backend
* Random seed
* Generation settings
* Available GPU memory

For deterministic evaluation, users should preserve the random seeds and decoding configurations specified in each notebook.

Quantized models may also produce small numerical differences across hardware and software environments.

---

## Hardware Requirements

The required hardware depends on the model and compression stage.

General recommendations:

* A CUDA-compatible GPU
* At least 24 GB of GPU memory for many full-model inference experiments
* More memory for continual pre-training or supervised fine-tuning
* Multi-GPU execution for larger training configurations
* Reduced batch sizes or gradient accumulation when memory is limited

INT8 and INT4 experiments require significantly less memory than full-precision inference.

---

## Data Availability

The repository provides experiment code and result files.

The underlying ArabLegalEval data may be subject to its original access, licensing, or redistribution conditions. Users should obtain the benchmark through its official distribution channel and place the required files in the paths specified in the notebooks.

No private credentials or restricted source documents are included in this repository.

---

## Checkpoints

Model checkpoints may be stored using Git Large File Storage when included.

To download LFS-tracked files:

```bash
git lfs install
git lfs pull
```

Some checkpoints may be omitted because of repository-size restrictions or model-license requirements. In such cases, the notebooks provide the training and model-loading procedures needed to reproduce them.

---

## Anonymous Review

This repository is prepared for anonymous peer review.

Identifying information, including author names, institutional affiliations, personal email addresses, and private service credentials, has been removed from the repository contents and Git commit metadata.

Please do not attempt to identify the authors during the review process.

---

## Citation

Citation information will be added after the paper is accepted.

```bibtex
@article{anonymous2026arabiclegalcompression,
  title   = {Three-Stage Compression for Arabic Legal Language Models},
  author  = {Anonymous Authors},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

The license for the code, model outputs, and derived artifacts will be finalized upon publication.

Third-party models, datasets, and libraries remain subject to their respective licenses and terms of use.

```
```
