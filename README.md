This repository contains code and datasets from the paper 'Help! Need advice on identifying Advice' to be presented at EMNLP 2020. If you found this paper useful, please consider citing our paper:

```
@inproceedings{venkat-etal-advice2020,
  author    = {Govindarajan, Venkata S  and Chen, Benjamin T and Warholic, Rebecca and  Li, Junyi Jessy  and  Erk, Katrin},
  title     = {Help! {N}eed {A}dvice on {I}dentifying {A}dvice},
  booktitle      = {Proceedings of The 2020 Conference on Empirical Methods in Natural Language Processing},
  year           = {2020},
}
```

## Abstract

Humans use language to accomplish a wide variety of tasks &mdash; asking for and giving advice being one of them. In online  advice  forums, advice is mixed in with non-advice, like emotional support, and is sometimes stated explicitly, sometimes implicitly. Understanding the language of advice would equip systems with a better grasp of language pragmatics; practically, the ability to identify advice would drastically increase the efficiency of advice-seeking online, as well as advice-giving in natural language generation systems.

We present a dataset in English from two Reddit advice forums &mdash; r/AskParents and r/needadvice &mdash; annotated for whether sentences in posts contain advice or not. Our analysis reveals rich linguistic phenomena in advice discourse. We present  preliminary models showing that while pre-trained language models are able to capture advice better than rule-based systems, advice identification is challenging, and we identify directions for future research.


## Recreating Python Environment

The experiments were carried out on a Linux machine with 4 GTX1080 graphics cards. We used [miniconda][conda] 4.8.2 to create a virtual python environment to perform all the modelling experiments. To recreate the same virtual environment, run:

`
conda env create --file environment.yml -n ENVIRONMENT_NAME
`

using the `environment.yml` file above. Note that the build environment is specifically for Linux. `environment_nobuild.py` is a platform agnostic build file, but some packages (like `cudatoolkit`) are not available on macOS.

## Reproducing Results

The python script `Advice_Classification_Simple.py` was used to produce the results in the paper. The bash script `train.sh` should train all model permutations used in various experiments. `test.sh` should reproduce the results of Table 6 in the paper.

To train a single model, use the following command:

```
python Advice_Classification_Simple.py --data DATASET --model MODEL --multigpu --seed SEED [--query] [--context] [--noft] [--frac 0-1]
```

The commandline arguments do the following:

 - `DATASET`: Should be either `askparents` or `needadvice`. Ensure that the dataset is in a folder called `Dataset` in the same directory.
 - `MODEL`: can take one of the following values - `bert`, `xlnet`, `roberta`, or `albert`. To reproduce results in the paper, pass `bert`.
 - `--multigpu`: Enables distributed training over all possible GPUs available.
 - `SEED`: Set the random seed.
 - `--query`: Augment sentences with query as described in paper and train
 - `--context`: Augment sentences with context.
 - `noft`: Set learning rate to 0 for all tramsformer layers except final classification layer.
 - `frac`: Fraction of training data to use (must be between 0 and 1). This is useful for transfer learning experiments

You can also specify learning rates for the classifier and transformer layers, weight decay, batch size, and dropout probability in transformer layers via commandline arguments.

To predict results on a test set, use the following command:

```
python Advice_Classification_Simple.py --test --data DATASET--model MODEL --multigpu --seed SEED --savedmodel PATH_TO_SAVED_MODEL_DIR/
```

<!-- We have provided two saved models &mdash; one for each dataset. Note that models are saved using the [`saved_pretrained`][savep] method from [Transformers][transf]. -->

<!-- Links -->

[conda]: https://docs.conda.io/en/latest/miniconda.html
<!-- [paper]: https://online.html -->
[savep]: https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.save_pretrained
[transf]: https://huggingface.co/transformers/index.html