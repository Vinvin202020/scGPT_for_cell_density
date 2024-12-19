# Cell Density Prediction Using scGPT

This repository contains the codebase for **scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI**. The project leverages scGPT to explore its applicability in spatial transcriptomics tasks such as cell density estimation.

### Online Applications of scGPT
scGPT can be used through browser-based apps hosted with cloud GPUs. These include:
- Reference mapping
- Cell annotation
- Gene Regulatory Network (GRN) inference

---

### Installation

scGPT requires **Python ≥ 3.7.13** and **R ≥ 3.6.1**. Please ensure you have the appropriate versions installed before proceeding with the installation.

#### Installing scGPT
scGPT is available on PyPI and can be installed using the following command:
```bash
pip install scgpt "flash-attn<1.0.5"  # optional, recommended
# As of 2023.09, pip install may not run with new versions of the google orbax package, if you encounter related issues, please use the following command instead:
# pip install scgpt "flash-attn<1.0.5" "orbax<0.1.8"
```
You can also install the required modules by using the **requirements.txt** file located in the docs directory.

We recommend using wandb for logging and visualization. To install it, run:
```python
pip install wandb
```

For developing, we are using the Poetry package manager. To install Poetry, follow the instructions here.

```bash
$ git clone this-repo-url
$ cd scGPT
$ poetry install
```
### Pretrained scGPT Model zoo

---

Below is a the link of the pretrained model. 

https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y

Model files should placed in the following directory:

scGPT_for_cell_density/save/scGPT_human
### Fine-tuning spatial Dataset for scGPT 

---

You can find the dataset used to fine-tune the model for cell density prediction on the Xenium page.

https://www.10xgenomics.com/datasets/preview-data-ffpe-human-lung-cancer-with-xenium-multimodal-cell-segmentation-1-standard

Dataset files should be placed in the following directory:

scGPT_for_cell_density/run/data
### Tasks/Milestones:
- Get familiar with transcriptomics concepts and data :
  This task involves understanding the core concepts of transcriptomics and exploring the data associated with spatial transcriptomics, which is crucial for interpreting the results.
- Understand scGPT model and code :
  The next milestone is to dive into the scGPT model, which was developed to work on single-cell transcriptomics data. However, this step presented significant challenges as the code was initially conceived for Linux, and we had to make several modifications to adapt it to our working environment. The process of understanding the model’s architecture and its codebase was quite complex and frustrating, as the code was neither well-documented nor straightforward to comprehend.
- Set up the environment, run the code, and explore the tutorial notebooks provided by scGPT :
  Setting up the environment for scGPT was an absolute pain. The dependencies were tricky to manage, and the environment setup took a lot of time. In order to run the code in a reasonable amount of time, we needed to ensure CUDA was properly configured, as the code heavily relies on GPU acceleration. Despite these challenges, we were able to get the environment up and running after considerable effort. Afterward, we explored the tutorial notebooks to understand the general workflow and how to apply the model.
- Implement cell density prediction and neighbors' cell type prediction from scGPT representations:
   After getting the environment ready, the next task involved implementing cell density prediction using the scGPT representations. We decided to compute cell density based on the spatial coordinates within the transcriptomics data. The design of the prediction head was another important step, and we had to carefully choose an appropriate architecture to handle the complexity of the data.
- Decide on a way to compute cell density based on the coordinates within the spatial transcriptomics data :
   This step required us to figure out how to compute the cell density from the given coordinates. It involved working with the spatial aspect of the transcriptomics data, which added another layer of complexity to the process.
- Decide on an architecture for the prediction head :
  For the cell density prediction task, we had to decide on a suitable neural network architecture for the prediction head. This required experimenting with various designs to find one that could best handle the task while maintaining reasonable computational efficiency.
- Train the prediction head on the cell density prediction task :
  After determining the architecture, the next milestone was to train the prediction head on the cell density task. This was an iterative process that required fine-tuning and adjusting the model to achieve good performance.
- Compare predictions to baseline :
  We then compared our model's predictions to a baseline. This helped us assess the effectiveness of our model by providing a point of reference for evaluation.
- Summarize the model and your cell density prediction outcomes in a report
### Notebook Organisation
---
- run/**Cell_density_regression** : Full notebook demonstrating the use of the scGPT pipeline to predict cell density through a regression approach.
- run/**Cell_density_classif** : Incomplete notebook where the cell density problem is approached as a classification task.
- run/draft/**First_glimpse_at_the_data** :  Notebook for initial exploration of the spatial dataset.
### Notes
---

### Acknowledgements
---
- scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI
- 10X genomics
- chatGPT for annotation of code

