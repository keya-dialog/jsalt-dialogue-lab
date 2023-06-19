# Neural Conversational AI Lab (JSALT 2023)
The lab will familiarize you with response generation for task-oriented dialogues (TOD) using end-to-end approaches.
We will use the MultiWOZ 2.2[ [1](https://arxiv.org/pdf/1810.00278.pdf), [2](https://aclanthology.org/2020.nlp4convai-1.13/)]🧙 dataset and causal language models implemented  the `huggingface/transformer` for a conditional generation.
The [QLoRa](https://arxiv.org/abs/2305.14314) implementation from `huggingface/peft` library will allow us to finetune large pretrained Large Langauge Models (LLMS) e.g.  LLAMA 🦙 and Falcon, on relatively small GPUs in Google Colab Notebook or on your cluster.

**What will you learn?**
- How to finetune large language model (LLM) using [QLoRa](https://huggingface.co/blog/4bit-transformers-bitsandbytes)💡 
- How to finetune parameters for [decoding/generation](https://huggingface.co/docs/transformers/main_classes/text_generation) with HuggingFace LLMs 🤗
- Get familiar with a typical TOD textual dataset MultiWoz[ [1](https://arxiv.org/pdf/1810.00278.pdf), [2](https://aclanthology.org/2020.nlp4convai-1.13/)].
- How to evaluate task-oriented dialogues (TOD) using [standardize scripts](https://github.com/Tomiinek/MultiWOZ_Evaluation).

 
 We prepared for you a series of tasks. A ready-to-use solution accompanies each task.
 The solutions are intentionally hidden, so you have the chance to try to work on the task on your own.
We are interested in how you use the code, so submit the answers via PRs or otherwise.
Pick your rewards 🍇!

## Environment setup

We prepared a `qlora.py` main python script and several bash launch scripts which showcase the `qlora.py` functionality.
The same functionality is demonstrated in a [Google Colab notebook](TODO).
The Google Colab is arguably more straightforward to set up but harder to work with.

#### Running on a GPU machine/cluster
If you have a machine with a recent GPU with 16GB of memory, we recommend creating a conda environment 
and installing the complete list of dependencies specified in `environment.yml`.

<details>

```bash
# Have a look at the environment.yml
# The QLoRa finetuning requires cutting-edge libraries versions
# Note: please use conda deactivate if you have other environment activated
#   sometimes it creates problems.
conda env create --prefix ./env -f environment.yml  # grab a coffee 

# activating the locally stored environment is easy
conda activate ./env

# Run the next turn prediction with the "debug" model argument argument. 
# It should trigger downloading a small pretrained model and the MultiWoz dataset from HuggingFace.
./scripts/finetune_multiwoz22_conditional_mlm.sh debug
```

</details>

**Task 1: Questions**
- How to run this script on the JSALT cluster? 🍇🍇🍇🍇
- What is your iteration speed for the training with the default values? 🍇
- What is your iteration speed for the inference speed with the default values? 🍇
- What machine and CUDA version do you have? 🍇
- How to run this script on the JSALT cluster? Contributions are welcome! 🍇🍇🍇🍇

**Task 1: Results**
Feel free to fill in partial information, e.g., if you do not know your CUDA version, just write '-'.

<details>

|GPU model |  CUDA   |  train [it/s]  | infer [it/s] |
|----------|---------|----------------|--------------|
|  waiting |  for    |    your        |  numbers     |

</details>

####  Google Colab

Open the [Google Colab](TODO).
Run the whole notebook and write down which GPU you were assigned and how much memory you have available.
The first dummy training should take around 20 minutes.
The script downloads a small pretrained model and the MultiWoz dataset from HuggingFace.

### Task 1: Questions
- What is your iteration speed for the training with the default values? 🍇
- What is your iteration speed for the inference speed with the default values? 🍇
- What machine and CUDA version do you have? 🍇🍇
- Can you get free machine with a GPU RAM larger than 16GB e.g. on Kaggle? 🍇🍇🍇🍇

**Please fill the `Task 1: Results` in the section for running on cluster. In the column `GPU model` prefix the GPU type with `GC`.**





## 🚀 Evaluating pretrained model
Let us start by comparing an untuned LLM (LLAMA) and an already fined-tuned `oplatek/llama-7b-todo-multi-woz` which I fine-tuned for you. (You will finetune your adapter/LoRa weights in the next task.) 

<details>

- Let's use the next turn generation, conditioned on previous dialogue context using the `./scripts/generate_prompted.sh` script.
- However the script is prepared to load the base model in 4bit but also the additional trained weights from the LoRa trained checkpoint.
- We do not have the LoRa checkpoint trained yet, so we need to modify the script.
- Copy the script

```bash
cp ./scripts/generate_prompted.sh ./scripts/pp.sh  # prompted_pretrained
```


- Open the `pp.sh`script and remove the `--checkpoint_dir "$checkpoint_dir"` line.
- Also adjust the `output_dir` to be named `output/$model_name_or_path/REST_IS_THE_SAME`
- The results should look like

```bash
  qlora.py \
    --dataloader_num_workers 0 \
    --max_eval_samples 1000 \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir "output/huggyllama/llama-7b/pred_multi_woz_v22_turns_1000_$$" \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --predict_with_generate \
    --per_device_eval_batch_size 4 \
    --dataset $dataset \
    --dataset_format $dataset_format \
    --source_max_len 256 \
    --target_max_len 288 \
    --max_new_tokens 32 \
    --do_sample \
    --top_p 0.9 \
    --num_beams 1 \
```

- Note that setting dataloader_num_workers to `0` is good for debugging. The dataloader runs in the main python thread. However, it is good to use more CPUs per 1 GPU if you are not debugging. 
- Explore the options and `qlora.py` especially the [Generation arguments](ttps://huggingface.co/docs/transformers/main_classes/text_generation). You can easily add them to the command line.

</details>

Play with parameters like `top_k`, `temperature`, `max_new_tokens`, `penalty_alpha`, etc.
Investigate [different decoding strategies](https://huggingface.co/docs/transformers/generation_strategies#contrastive-search).

### Task 3: Questions
- What is the highest `batch_size` you can use for decoding with otherwise default values? 🍇
- What is the longest reply you can force the model to generate with default values? 🍇🍇 
- How can you force the code to behave deterministically when having the same dialogue history and already fixed random seed? 🍇🍇🍇
- Best bleu, success, inform, richness score without fine tuning?

### Task 3: Results
<details>

|LLM model |  Decoding params |  Bleu  |   Success | Inform |  Richness |
|----------|------------------|--------|-----------|--------|-----------|
|  waiting |  for             |   your |  numbers  | again  |           |

</details>



## 💪 Finetune LLAMA with QLora
Finally! Let us train the LoRa weights!

<details>
- Easy:)

```bash
./scripts/finetune_multiwoz22_conditional_mlm.sh huggyllama/llama-7b
```
- However, you may want to start small. So also explore small models `EleutherAI/pythia-70m`, set number of training steps to much lower number, etc.
- Warning: see how checkpoint works. Adjust `save_steps` so you will have at least some checkpoint after training.

</details>

### Task 4: Questions
- What LoRa modules work best? `attention`, `ffn`, `regexp_keys|values`, ...? 🍇🍇🍇
- For the default parameters, what is the best number of training steps?🍇🍇
- What is the best learning rate and number of training steps?🍇🍇🍇
- Can you implement prompting to generate the conversation of certain length?🍇🍇🍇🍇🍇
  - Hint: I would start with `multi_woz_v22_dialogs` format as used in `finetune_multiwoz22_standard_mlm.sh`.
  - The `multi_woz_v22_turns` format always "prompts" the model with dialogue history ending with `...\nbot>` telling the model to reply as a bot.
  - The `multi_woz_v22_turns` format is used in `scripts/finetune_multiwoz22_conditional_mlm.sh`

### Task 4: Results

<details>

|LLM model |  Training params |  Bleu  |   Success | Inform |  Richness |
|----------|------------------|--------|-----------|--------|-----------|
|  waiting |  for             |   your |  numbers  | again  |           |

</details>


## 🏆 Explore available pretrained LLMs  

Open the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) and try to run different models.
The LLAMA models and their derivations, such as Alpaca and Vicuna, should be compatible with the script.
We also tested the code with `EleutherAI/pythia-70m`.
Try also to scale the models' size, e.g., `EleutherAI/pythia-12b` instead `EleutherAI/pythia-70m`.
Note that the `pythia-70m`` model is excellent for debugging.
Try also models trained on different datasets `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`.

### Task5: Questions
- Do zero-shot models perform better as the number of parameters grows? For which metrics? 
  - Report results with `huggyllama/llama*` or `EleutherAI/pythia*` checkpoints. 🍇🍇
  - For other models, try at least three different sizes for the same model.  🍇🍇🍇
- What is the largest model you were able to finetune? 🍇

**Please, insert the answers into Task 3: Results table.**


## ✅︎ Bored? Improve the code! ✅︎

_Please open a Pull Request._

- Add the possibility to add an "instruction" prompt before dialogue history🍇🍇🍇
- Implement Evaluation callback to evaluate regularly during training.🍇🍇🍇
- Train from scratch using `full_finetune` and reinitializing the weights](https://github.com/J4VORSKY/JSALT2023-MT-lab/blob/main/solutions/task_6.py#L26) with reasonable hyperparameters.🍇🍇🍇🍇
- Add `span_info` to the dataloader and tag named entities.🍇🍇🍇🍇.
- Add dialogue state information to the dataloader and predict dialogue state instead of the words of the next response.🍇🍇🍇🍇🍇.
- Clean the code 🍇

## 💡 Up for a challenge? What is the minimal experiment to add?💡
_Below is a conversation starter list related to the topic of [clustering dialogues](https://github.com/keya-dialog/jsalt-dialogue-lab)._

- Evaluate Encoder-Decoder models, e.g., Flan-T5. Which architecture is better? [Up for debate/experiments.](https://twitter.com/ShayneRedford/status/1668720485285199872?t=f3I3FS2VZ9Woq7GuyOeosg&s=19). 🤷
- How would you use the embeddings for evaluation?
- Do you use conversational models? How would you like to evaluate them?
- What should be the first experiment for multimodal setup with audio/speech and text?
- Can you recommend some spoken task-oriented datasets? We know about:
    - [Let's go](https://github.com/DialRC/LetsGoDataset)
    - [SpokenWOZ](https://spokenwoz.github.io/SpokenWOZ-github.io/) 
    - [Ryan speech](http://mohammadmahoor.com/ryanspeech-request-form/)
    - [DSTC11: Speech-Aware Dialog Systems Technology Challenge](https://storage.googleapis.com/gresearch/dstc11/dstc11.2022-09-29a.html)
    - ❓
- Is there a dataset that captures prosody similar to a call-center agent / ideal prosody for a dialogue system?
- [How to improve embeddings for automatic clustering](https://www.clsp.jhu.edu/ai_research_internships_for_undergraduates_23/#autodesign)?

## 👏 Contributing

If you have implemented a new feature, found a bug, or want to fix a typo, please submit a pull request.🙏 

Use the [black](https://github.com/psf/black) formatter to avoid merge conflicts in large PRs.

In other cases, feel free to reach us too:<br/>
[Ondřej Plátek](opla.cz), [(UFAL, Charles University, Prague)](https://ufal.mff.cuni.cz/ondrej-platek) <br/>
[Santosh Kesiraju](https://www.fit.vut.cz/person/kesiraju/.cs), [(FIT, VUT, Brno)](https://www.fit.vut.cz/person/kesiraju/) <br/>
[Petr Schwarz](https://www.fit.vut.cz/person/schwarzp/.en), [(FIT, VUT, Brno)](https://www.fit.vut.cz/person/schwarzp/) <br/>

## 💭 Citation
If you use the code or results from this tutorial, please cite the tutorial in the following manner:
```
@article{oplatek2023qlora-multiwoz,
  title={Investigating Masked Language Model and Instruction finetuning of LLMs using QLoRa for Task-Oriented Dialogue Models},
  author={Plátek, Ondřej and Kesiraju, Santosh and Schwarz, Petr},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/keya-dialog/jsalt-dialogue-lab}},
  commit = {todo}
  year={2023}
}
```

Please, also cite the [artidoro/qlora](https://github.com/artidoro/qlora) project on which our work is built on.
```
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```
