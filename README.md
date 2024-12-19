scGPT
This is the official codebase for scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI.

Preprint   Documentation   PyPI version   Downloads   Webapp   License

!UPDATE: We have released several new pretrained scGPT checkpoints. Please see the Pretrained scGPT checkpoints section for more details.

[2024.02.26] We have provided a priliminary support for running the pretraining workflow with HuggingFace at the integrate-huggingface-model branch. We will conduct further testing and merge it to the main branch soon.

[2023.12.31] New tutorials about zero-shot applications are now available! Please see find them in the tutorials/zero-shot directory. We also provide a new continual pretrained model checkpoint for cell embedding related tasks. Please see the notebook for more details.

[2023.11.07] As requested by many, now we have made flash-attention an optional dependency. The pretrained weights can be loaded on pytorch CPU, GPU, and flash-attn backends using the same load_pretrained function, load_pretrained(target_model, torch.load("path_to_ckpt.pt")). An example usage is also here.

[2023.09.05] We have release a new feature for reference mapping samples to a custom reference dataset or to all the millions of cells collected from CellXGene! With the help of the faiss library, we achieved a great time and memory efficiency. The index of over 33 millions cells only takes less than 1GB of memory and the similarity search takes less than 1 second for 10,000 query cells on GPU. Please see the Reference mapping tutorial for more details.

Online apps
scGPT is now available at the following online apps as well, so you can get started simply with your browser!

Run the reference mapping app, cell annotation app and the GRN inference app with cloud gpus. Thanks to the Superbio.ai team for helping create and host the interactive tools.
Installation
scGPT works with Python >= 3.7.13 and R >=3.6.1. Please make sure you have the correct version of Python and R installed pre-installation.

scGPT is available on PyPI. To install scGPT, run the following command:

pip install scgpt "flash-attn<1.0.5"  # optional, recommended
# As of 2023.09, pip install may not run with new versions of the google orbax package, if you encounter related issues, please use the following command instead:
# pip install scgpt "flash-attn<1.0.5" "orbax<0.1.8"
[Optional] We recommend using wandb for logging and visualization.

pip install wandb
For developing, we are using the Poetry package manager. To install Poetry, follow the instructions here.

$ git clone this-repo-url
$ cd scGPT
$ poetry install
Note: The flash-attn dependency usually requires specific GPU and CUDA version. If you encounter any issues, please refer to the flash-attn repository for installation instructions. For now, May 2023, we recommend using CUDA 11.7 and flash-attn<1.0.5 due to various issues reported about installing new versions of flash-attn.

Pretrained scGPT Model Zoo
Here is the list of pretrained models. Please find the links for downloading the checkpoint folders. We recommend using the whole-human model for most applications by default. If your fine-tuning dataset shares similar cell type context with the training data of the organ-specific models, these models can usually demonstrate competitive performance as well. A paired vocabulary file mapping gene names to ids is provided in each checkpoint folder. If ENSEMBL ids are needed, please find the conversion at gene_info.csv.

Model name	Description	Download
whole-human (recommended)	Pretrained on 33 million normal human cells.	link
continual pretrained	For zero-shot cell embedding related tasks.	link
brain	Pretrained on 13.2 million brain cells.	link
blood	Pretrained on 10.3 million blood and bone marrow cells.	link
heart	Pretrained on 1.8 million heart cells	link
lung	Pretrained on 2.1 million lung cells	link
kidney	Pretrained on 814 thousand kidney cells	link
pan-cancer	Pretrained on 5.7 million cells of various cancer types	link
Fine-tune scGPT for scRNA-seq integration
Please see our example code in examples/finetune_integration.py. By default, the script assumes the scGPT checkpoint folder stored in the examples/save directory.

To-do-list
 Upload the pretrained model checkpoint
 Publish to pypi
 Provide the pretraining code with generative attention masking
 Finetuning examples for multi-omics integration, cell type annotation, perturbation prediction, cell generation
 Example code for Gene Regulatory Network analysis
 Documentation website with readthedocs
 Bump up to pytorch 2.0
 New pretraining on larger datasets
 Reference mapping example
 Publish to huggingface model hub
Contributing
We greatly welcome contributions to scGPT. Please submit a pull request if you have any ideas or bug fixes. We also welcome any issues you encounter while using scGPT.

Acknowledgements
We sincerely thank the authors of following open-source projects:

flash-attention
scanpy
scvi-tools
scib
datasets
transformers
Citing scGPT
@article{cui2023scGPT,
title={scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI},
author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Wang, Bo},
journal={bioRxiv},
year={2023},
publisher={Cold Spring Harbor Laboratory}
}
