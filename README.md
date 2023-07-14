# followQG

This repo contains the code accompanying the paper [Automatic Follow-up Question Generation for Asynchronous Interviews](https://intellang.github.io/papers/3-IntelLanG_2020_paper_3.pdf) published in IntelLang workshop of ECAI 2020 and its extended version [Improving Asynchronous Interview Interaction with Follow-up Question Generation](https://dx.doi.org/10.9781/ijimai.2021.02.010).

This codebase contains training and testing scripts along with the in-domain interview follow-up question dataset that can be used to train a follow-up question generation model. 

## Installation

To install and use the training and inference scripts please clone the repo and install the requirements:

```bash
git clone https://github.com/poorao/followQG.git
cd followQG
pip install -r requirements.txt
python -m spacy download en
```

You can then run the `interact.py` script on the pretrained model:

```bash
python interact.py --dataset_path ../followML_candidates.json --model gpt2 --model_checkpoint runs/Jul03_10-47-53_ip-172-31-11-159 --top_k 10
```

## Pretrained model

We make a pretrained and fine-tuned model available on our S3.

## Using the training script

```bash
python train.py --dataset_path ../followML_candidates.json --model_checkpoint gpt2 --train_batch_size 2 --valid_batch_size 2  # Single GPU training
```

## Using the interaction script

The training script saves all the experiments and checkpoints in a sub-folder named with the timestamp of the experiment in the `./runs` folder of the repository base folder.

You can then use the interactive script to interact with the model simply by pointing to this folder.

Here is an example command line to run the interactive script:

```bash
python interact.py --dataset_path ../followML_candidates.json --model gpt2 --model_checkpoint runs/Jul03_10-47-53_ip-172-31-11-159 --top_k 10  # run the interactive script with a training checkpoint
```

The interactive script accept a few arguments to tweak the decoding algorithm:

Argument | Type | Default value | Description
---------|------|---------------|------------
dataset_path | `str` | `""` | Path or url of the dataset. If empty download from S3.
dataset_cache | `str` | `'./dataset_cache.bin'` | Path or url of the dataset cache
model | `str` | `"openai-gpt"` | Path, url or short name of the model
max_history | `int` | `2` | Number of previous utterances to keep in history
device | `str` | `cuda` if `torch.cuda.is_available()` else `cpu` | Device (cuda or cpu)
no_sample | action `store_true` | Set to use greedy decoding instead of sampling
max_length | `int` | `20` | Maximum length of the output utterances
min_length | `int` | `1` | Minimum length of the output utterances
seed | `int` | `42` | Seed
temperature | `int` | `0.7` | Sampling softmax temperature
top_k | `int` | `0` | Filter top-k tokens before sampling (`<=0`: no filtering)
top_p | `float` | `0.9` | Nucleus filtering (top-p) before sampling (`<=0.0`: no filtering)

## Citation

If you use this code in your research, you can cite our IntelLang workshop [paper](https://intellang.github.io/papers/3-IntelLanG_2020_paper_3.pdf):

```bash
@article{2902,
  keywords = {Asynchronous Video Interview, Language Model, Question Generation, Conversational Agent, Follow-up Question Generation},
  author = {Pooja S. B. Rao and Manish Agnihotri and Dinesh Babu Jayagopi},
  title = {Improving Asynchronous Interview Interaction with Follow-up Question Generation},
  abstract = {The user experience of an asynchronous video interview system, conventionally is not reciprocal or conversational. Interview applicants expect that, like a typical face-to-face interview, they are innate and coherent. We posit that the planned adoption of limited probing through follow-up questions is an important step towards improving the interaction. We propose a follow-up question generation model (followQG) capable of generating relevant and diverse follow-up questions based on the previously asked questions, and their answers. We implement a 3D virtual interviewing system, Maya, with capability of follow-up question generation. Existing asynchronous interviewing systems are not dynamic with scripted and repetitive questions. In comparison, Maya responds with relevant follow-up questions, a largely unexplored feature of irtual interview systems. We take advantage of the implicit knowledge from deep pre-trained language models to generate rich and varied natural language follow-up questions. Empirical results suggest that followQG generates questions that humans rate as high quality, achieving 77% relevance. A comparison with strong baselines of neural network and rule-based systems show that it produces better quality questions. The corpus used for fine-tuning is made publicly available.},
  year = {2021},
  journal = {International Journal of Interactive Multimedia and Artificial Intelligence},
  volume = {6},
  number = {5},
  pages = {79-89},
  month = {03/2021},
  issn = {1989-1660},
  url = {https://www.ijimai.org/journal/sites/default/files/2021-02/ijimai_6_5_8.pdf},
  doi = {10.9781/ijimai.2021.02.010},
}

```

## Reference

This repository is based on [transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai) from [HuggingFace](https://github.com/huggingface).
