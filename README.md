# Cell Density Prediction Using scGPT

This repository contains the codebase for **scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI**. The project leverages scGPT to explore its applicability in spatial transcriptomics tasks such as cell density estimation.

### Online Applications
You can now use scGPT through browser-based apps hosted with cloud GPUs. These include:
- Reference mapping
- Cell annotation
- Gene Regulatory Network (GRN) inference

---

### Installation

scGPT requires **Python ≥ 3.7.13** and **R ≥ 3.6.1**. Please ensure you have the appropriate versions installed before proceeding with the installation.

Installing scGPT
scGPT is available on PyPI and can be installed using the following command:
asdasd

You can also install the required modules by using the **requirements.txt** file located in the docs directory.

bash
pip install scgpt "flash-attn<1.0.5"  # optional, recommended
Troubleshooting Installation Issues
As of September 2023, newer versions of the google-orbax package may cause conflicts. If you encounter issues, use this alternative command:


pip install scgpt "flash-attn<1.0.5" "orbax<0.1.8"
Optional: Logging and Visualization
We recommend using wandb for logging and visualization. To install it, run:


pip install wandb

### Pretrained scGPT Model zoo

---

Below is a the link of the pretrained model. 

### Fine-tuning spatial Dataset for scGPT 

---

You can find the dataset used to fine-tune the model for cell density prediction on the Xenium page.



### Acknowledgements

---

