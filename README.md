# BCI Project: Phoneme-to-Text Decoding
### Co-Developed by Tony Korol, Ted Jang, Kushagra Tiwari

This repository contains the codebase for our BCI Neural Speech Decoder. Our work focuses on decoding neural speech spike-rasters into standard English text using a combination of traditional neural architectures (GRU) and fine-tuned Large Language Models (T5).

---

## Featured Resources

### 🤗 HuggingFace Fine-Tuned LLM
We have fine-tuned a T5 model specifically for phoneme-to-text transcription. The model weights and tokenizer are available for public use:
* **[tonykorol/t5_phoneme_decoder](https://huggingface.co/tonykorol/t5_phoneme_decoder)**

### Weights and Biases (W&B) Training Logs
Explore our comprehensive training sessions, loss curves, and performance metrics for both our GRU and LLM trials:
* **[BCI Final Project: GRU & Comparison Trials](https://wandb.ai/tkorol1-ucla/BCI%20Final%20Project?nw=nwusertkorol1)**
* **[HuggingFace T5 Fine-Tuning Logs](https://wandb.ai/tkorol1-ucla/huggingface?nw=nwusertkorol1)**
> *Note: GRU logging may appear fragmented due to training interruptions and restarts.*

---

## Quick Start: Running the Demo

If you wish to test the fine-tuned T5 phoneme decoder's performance immediately without setting up the full 3.6 GB dataset, follow these steps:

1. **Open the Demo Notebook**: Locate and open **`PtoT_Final_Demo_cleaned.ipynb`**.
2. **Standalone Execution**: This notebook is "light-weight" and does not require external data files. It contains a built-in `SAMPLES` list of phoneme sequences.
3. **Randomized Testing**: The demo script is configured to select a **random index** from the sample list each time it is executed, providing a fresh demonstration of the T5 model's decoding and Word Error Rate (WER) performance.



---

## Repository Structure

This repository includes several files for training, processing, and deployment:

### Training & Development Notebooks
* **`Phoneme_to_Text_cleaned.ipynb`**: Training Notebook for t5 phoneme to text fine-tuning
* **`Revised_train_model_cleaned.ipynb`**: Optimized training routine featuring text-to-phoneme conversion.
* **`train_model_cleaned.ipynb`**: Baseline model implementation.

### Core Scripts
* **`model.py`**: .py file with optimal model parameters
* **`neural_decoder_trainer.py`**: Contains the underlying training logic and utility functions.
