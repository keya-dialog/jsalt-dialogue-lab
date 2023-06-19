# Neural Conversational AI Lab (JSALT 2023)
The lab will familiarize you with response generation for task-oriented dialogues (TOD) using end-to-end approaches.
We will use the MultiWOZ 2.2[1](https://arxiv.org/pdf/1810.00278.pdf)[2](https://aclanthology.org/2020.nlp4convai-1.13/)🧙 dataset and causal language models implemented in the `huggingface/`transformer` for a conditional generation.
The QLoRa implementation from `huggingface/peft` library will allow us to finetune large pretrained Large Langauge Models (LLMS) e.g.  LLAMA 🦙 and Falcon, on relatively small GPUs in Google Colab Notebook or on your cluster.

**What will you learn?**
- How to finetune large language model (LLM)💡 using [QLoRa](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- How to finetune parameters for [decoding/generation](https://huggingface.co/docs/transformers/main_classes/text_generation) with HuggingFace LLMs 🤗
- Get familiar with a typical TOD textual dataset MultiWoz[1](https://arxiv.org/pdf/1810.00278.pdf)[2](https://aclanthology.org/2020.nlp4convai-1.13/)🧙 .
- How to evaluate task-oriented dialogues (TOD) using [standardize scripts](https://github.com/Tomiinek/MultiWOZ_Evaluation)

 
 We prepared for you a series of tasks. A ready-to-use solution accompanies each task.
 The solutions are intentionally hidden, so you have the chance to try to work on the task on your own.
We are interested in how you use the code so submit the answers via PRs or otherwise.
Pick your rewards 🍇!

## Environment setup

We prepared a `qlora.py` main python script and several bash launch scripts which showcase the `qlora.py` functionality.
The same functionality is demonstrated in a [Google Colab notebook](TODO).
The Google Colab is arguably more straightforward to set up but harder to work with.

### Task 0

Set up your environment.

#### Running on a GPU machine/cluster
If you have a machine with a recent GPU with 16GB of memory, we recommend creating a conda environment 
and installing the complete list of dependencies specified in `environment.yml`.

<details>
bash
```
# Have a look at the environment.yml
# The QLoRa finetuning requires cutting-edge libraries versions
conda env create --prefix ./env -f environment.yml  # grab a coffee 

# activating the locally stored environment is easy
conda activate ./env

# Run the main with debug argument. 
# It should trigger downloading a small pretrained model and the MultiWoz dataset from HuggingFace.
TODO
```
</details>

**Questions**
- How to run this script on the JSALT cluster? 🍇🍇🍇🍇
- What is your iteration speed for the training with the default values? 🍇
- What is your iteration speed for the inference speed with the default values? 🍇
- What machine and CUDA version do you have? 🍇🍇

**Answers**
Feel free to fill in partial information e.g. if you do not know your CUDA version just write '-'.
<details>
| GPU model |  CUDA   |  train [it/s]  | infer [it/s] |
| ----------|---------|----------------|--------------|
|   waiting |  for    |    your        |  numbers     |
</details>

####  Google Colab

Open the [Google Colab](TODO).
Run the whole notebook and write down which GPU you were assigned and how much memory you have available.
The first dummy training should take around 20 minutes.
The script downloads a small pretrained model and the MultiWoz dataset from HuggingFace.


### Evaluating pretrained model
Let us start by comparing an untuned LLM (LLAMA) and an already fined-tuned LLAMA model using the functionality from the next section.


```
# Run the evaluation code for "huggyllama/llama-7b" and the "oplatek/llama-7b-mwz22-basic01".
TODO
``` 
Play with parameters like `top_k`, `temperature`, `max_new_tokens, `penalty_alpha`, etc.
Investigate [different decoding strategies](https://huggingface.co/docs/transformers/generation_strategies#contrastive-search).

**Question**
- What is the highest `batch_size` you can use for decoding with otherwise default values? 🍇
- What is the longest reply you can force the model to generate with `batch_size`? 🍇🍇 
- How can you force the code to behave deterministically when having the same dialogue history and already fixed random seed? 🍇🍇🍇
- Best bleu, success, inform, richness score without fine tuning?

**Results**
<details>
| LLM model |  Decoding params |  Bleu  |   Success | Inform |  Richness |
| ----------|------------------|--------|-----------|--------|-----------|
|   waiting |  for             |   your |  numbers  | again  |           |
</details>



### Finetune LLAMA with QLora

**Questions**
- What LoRa modules work best? `attention`, `ffn`, `regexp_keys|values`, ...? 🍇🍇🍇🍇🍇
- For the default parameters, what is the best number of training steps?🍇🍇🍇
- What is the best learning rate and number of training steps?🍇🍇🍇

**Results**
<details>
| LLM model |  Training params |  Bleu  |   Success | Inform |  Richness |
| ----------|------------------|--------|-----------|--------|-----------|
|   waiting |  for             |   your |  numbers  | again  |           |
</details>


### Explore of-the-shelf LLMs  

Open the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) and try to run different models.
The LLAMA models and their derivations, such as Alpaca and Vicuna, should be compatible with the script.
We also tested the code with `EleutherAI/pythia-70m`.
Try also to scale the models' size, e.g., `EleutherAI/pythia-12b` instead `EleutherAI/pythia-70m`.
Note that the `pythia-70m`` model is excellent for debugging.
Try also models trained on different datasets `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`.

**Please insert the answers into the results in the table above for results with different decoding strategies**

**Questions**
- Do zero-shot models perform better as the number of parameters grows? For which metrics? 
  - Report results with `huggyllama/llama*` or `EleutherAI/pythia*` checkpoints. 🍇🍇
  - For other models, try at least three different sizes for the same model.  🍇🍇🍇
- What is the largest model you were able to finetune? 🍇

#### What next? Fix known TODOs
- Add the possibility to add "instruction" prompt before dialogue history🍇🍇🍇
- Implement Evaluation callback to evaluate regularly during training.🍇🍇🍇
- Train from scratch using `full_finetune` and [reinitilizing the weights](https://github.com/J4VORSKY/JSALT2023-MT-lab/blob/main/solutions/task_6.py#L26) with reasonable hyperparameters.🍇🍇🍇🍇
- Clean the code 🍇

#### What next? Experiments?Research Ideas? 💡 
- Evaluate Encoder-Decoder models, e.g. Flan-T5. Which architecture is better? There is a lot of [confusion](https://twitter.com/ShayneRedford/status/1668720485285199872?t=f3I3FS2VZ9Woq7GuyOeosg&s=19). 🤷
- How would you use the embeddings for evaluation?
- Do you use conversational models? How would you like to evaluate them?
- Can you recommend us some spoken task-oriented datasets? We know about:
    - [Let's go](https://github.com/DialRC/LetsGoDataset)
    - [SpokenWOZ](https://spokenwoz.github.io/SpokenWOZ-github.io/) 
    - [Ryan speech](http://mohammadmahoor.com/ryanspeech-request-form/)
    - [DSTC11: Speech-Aware Dialog Systems Technology Challenge](https://storage.googleapis.com/gresearch/dstc11/dstc11.2022-09-29a.html)
    - ❓
- Is there a dataset that captures prosody similar to a call-center agent / ideal prosody for a dialogue system?
- [How to improve embeddings for automatic clustering](https://www.clsp.jhu.edu/ai_research_internships_for_undergraduates_23/#autodesign)?



## Contact
- [Ondřej Plátek](opla.cz), [(UFAL, Charles University, Prague)](https://ufal.mff.cuni.cz/ondrej-platek))
- [Santosh Kesiraju](https://www.fit.vut.cz/person/kesiraju/.cs), [(FIT, VUT, Brno)](https://www.fit.vut.cz/person/kesiraju/)
- [Petr Schwarz](https://www.fit.vut.cz/person/schwarzp/.en), [(FIT, VUT, Brno)](https://www.fit.vut.cz/person/schwarzp/)

## Contribute

If you have implemented a new feature, found a bug or want to fix a typo, please submit a pull request.🙏 

## Citation

```
If you use the code or results from this tutorial please cite the tutorial in the following manner:
@article{oplatek2023qlora-multiwoz,
  title={Investigating Masked Language Model and Instruction finetuning of LLMs using QLoRa for Task-Oriented Dialogue Models},
  author={Plátek, Ondřej and Kesiraju, Santosh and Schwarz, Petr},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/keya-dialog/jsalt-dialogue-lab}},
  commit = {todo}
  year={2023}
}

Please, also cite the project on which our work is built on:
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```