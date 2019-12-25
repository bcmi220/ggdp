# Global Greedy Dependency Parsing (ggdp)

Codes for the paper **Global Greedy Dependency Parsing** in AAAI 2020

## Update

Since the completion of this paper was earlier than the beginning of 2019, the paper was based on the implementation of LSTM + pre-trained embeddings (features extracted from pre-trained language model layers). Now the pre-training - fine-tuning mode is more popular, so I re-transplanted my model based on [transformers](https://github.com/huggingface/transformers), and achieved further improvements in results (including baseline), and I also reported the results on the various popular pre-training models (BERT, XLNet, Roberta etc.). Hope this will help you.

**「Happy parsing, happy life!」**

## Abstract

Most syntactic dependency parsing models may fall into one of two categories: transition- and graph-based models. The former models enjoy high inference efficiency with linear time complexity, but they rely on the stacking or reranking of partially-built parse trees to build a complete parse tree and are stuck with slower training for the necessity of dynamic oracle training. The latter, graph-based models, may boast better performance but are unfortunately marred by polynomial time inference. In this paper, we propose a novel parsing order objective, resulting in a novel dependency parsing model capable of both global (in sentence scope) feature extraction as in graph models and linear time inference as in transitional models. The proposed global greedy parser only uses two arc-building actions, left and right arcs, for projective parsing. When equipped with two extra non-projective arc-building actions, the proposed parser may also smoothly support non-projective parsing. Using multiple benchmark treebanks, including the Penn Treebank (PTB), the CoNLL-X treebanks, and the Universal Dependency Treebanks, we evaluate our parser and demonstrate that the proposed novel parser achieves good performance with faster training and decoding.

## Results

| Model with MST infer.             | UAS     | LAS     
| --------------------------------- | ------- | --------
| `bert-large-cased`                | 96.92   | 95.35   
| `xlnet-large-cased`               | 97.22   | 95.66   
| `roberta-large`                   | 97.11   | 95.51   
| `xlm-roberta-large`               | 97.06   | 95.56   
| `distilbert-base-uncased`         | 96.13   | 94.43   
| `albert-xxlarge-v1`               | 97.09   | 95.24   
| `albert-xxlarge-v2`               | 97.17   | 95.45   


| Model with GGDP projective infer. | UAS     | LAS     | Order Acc.     
| --------------------------------- | ------- | --------|-----------
| `bert-large-cased`                | 96.57   | 95.05   | 90.36
| `xlnet-large-cased`               | 96.97   | 95.37   | 91.59
| `roberta-large`                   |         |         |
| `xlm-roberta-large`               |         |         |
| `distilbert-base-uncased`         |         |         |
| `albert-xxlarge-v1`               |         |         |
| `albert-xxlarge-v2`               |         |         |


| Model with GGDP non-projective infer.  | UAS     | LAS     | Order Acc.     
| -------------------------------------- | ------- | --------|-----------
| `bert-large-cased`                     |  96.70  | 95.20   | 90.36
| `xlnet-large-cased`                    |  97.12  | 95.53   | 91.59
| `roberta-large`                        |         |         |
| `xlm-roberta-large`                    |         |         |
| `distilbert-base-uncased`              |         |         |
| `albert-xxlarge-v1`                    |         |         |
| `albert-xxlarge-v2`                    |         |         |


## Prepare
```bash
python ./examples/generate_postag_and_labels.py ./data/ptb3.3/ptb3.3-stanford.auto.cpos.train.conll ./data/ptb3.3/postag-{ptb3.3-stanford.auto.cpos}.txt ./data/ptb3.3/label-{ptb3.3-stanford.auto.cpos}.txt
```

## GGDP projective inference algorithm

### Train:
```bash
python ./examples/run_dependency_parsing_with_order.py \
    --data_dir ./data/ptb3.3 \
    --train_file ptb3.3-stanford.auto.cpos.train.conll \
    --eval_file ptb3.3-stanford.auto.cpos.dev.conll \
    --model_type bert \
    --model_name_or_path ./pretrain/bert/bert-large-cased/ \
    --output_dir ./outputs/dependency_parsing_with_order_ggp-bert-large-cased \
    --max_seq_length 256 \
    --num_train_epochs 10 \
    --per_gpu_train_batch_size 24 \
    --labels ./data/ptb3.3/label-{ptb3.3-stanford.auto.cpos}.txt \
    --logging_steps 50 \
    --convert_strategy 0 \
    --infer_alg ggp \
    --evaluate_during_training \
    --eval_strategy 2 \
    --eval_steps 100 \
    --save_strategy 1 \
    --seed 1 \
    --do_train \
    --do_eval
```

### Predict:
```bash
python ./examples/run_dependency_parsing_with_order.py \
    --data_dir ./data/ptb3.3 \
    --predict_file ptb3.3-stanford.auto.cpos.test.conll \
    --predict_output ptb3.3-stanford.auto.cpos.test.pred \
    --model_type bert \
    --model_name_or_path ./outputs/dependency_parsing_with_order_ggp-bert-large-cased/ \
    --infer_alg ggp \
    --output_dir ./outputs/ \
    --max_seq_length 256 \
    --labels ./data/ptb3.3/label-{ptb3.3-stanford.auto.cpos}.txt \
    --convert_strategy 0 \
    --do_predict
```

## GGDP non-projective inference algorithm

### Train:
```bash
python ./examples/run_dependency_parsing_with_order.py \
    --data_dir ./data/ptb3.3 \
    --train_file ptb3.3-stanford.auto.cpos.train.conll \
    --eval_file ptb3.3-stanford.auto.cpos.dev.conll \
    --model_type bert \
    --model_name_or_path ./pretrain/bert/bert-large-cased/ \
    --output_dir ./outputs/dependency_parsing_with_order_ggp-bert-large-cased \
    --max_seq_length 256 \
    --num_train_epochs 10 \
    --per_gpu_train_batch_size 24 \
    --labels ./data/ptb3.3/label-{ptb3.3-stanford.auto.cpos}.txt \
    --logging_steps 50 \
    --convert_strategy 0 \
    --infer_alg ggnp \
    --evaluate_during_training \
    --eval_strategy 2 \
    --eval_steps 100 \
    --save_strategy 1 \
    --seed 1 \
    --do_train \
    --do_eval
```

### Predict:
```bash
python ./examples/run_dependency_parsing_with_order.py \
    --data_dir ./data/ptb3.3 \
    --predict_file ptb3.3-stanford.auto.cpos.test.conll \
    --predict_output ptb3.3-stanford.auto.cpos.test.pred \
    --model_type bert \
    --model_name_or_path ./outputs/dependency_parsing_with_order_ggp-bert-large-cased/ \
    --infer_alg ggnp \
    --output_dir ./outputs/ \
    --max_seq_length 256 \
    --labels ./data/ptb3.3/label-{ptb3.3-stanford.auto.cpos}.txt \
    --convert_strategy 0 \
    --do_predict
```

## MST algorithm

### Train:
```bash
python ./examples/run_dependency_parsing_with_order.py \
    --data_dir ./data/ptb3.3 \
    --train_file ptb3.3-stanford.auto.cpos.train.conll \
    --eval_file ptb3.3-stanford.auto.cpos.dev.conll \
    --model_type bert \
    --model_name_or_path ./pretrain/bert/bert-large-cased/ \
    --output_dir ./outputs/dependency_parsing_with_order_ggp-bert-large-cased \
    --max_seq_length 256 \
    --num_train_epochs 10 \
    --per_gpu_train_batch_size 24 \
    --labels ./data/ptb3.3/label-{ptb3.3-stanford.auto.cpos}.txt \
    --logging_steps 50 \
    --convert_strategy 0 \
    --infer_alg mst \
    --evaluate_during_training \
    --eval_strategy 2 \
    --eval_steps 100 \
    --save_strategy 1 \
    --seed 1 \
    --do_train \
    --do_eval
```

### Predict:
```bash
python ./examples/run_dependency_parsing_with_order.py \
    --data_dir ./data/ptb3.3 \
    --predict_file ptb3.3-stanford.auto.cpos.test.conll \
    --predict_output ptb3.3-stanford.auto.cpos.test.pred \
    --model_type bert \
    --model_name_or_path ./outputs/dependency_parsing_with_order_ggp-bert-large-cased/ \
    --infer_alg mst \
    --output_dir ./outputs/ \
    --max_seq_length 256 \
    --labels ./data/ptb3.3/label-{ptb3.3-stanford.auto.cpos}.txt \
    --convert_strategy 0 \
    --do_predict
```

## Reference

Please kindly cite this paper in your publications if it helps your research:

```
@inproceedings{li2020ggdp,
	title={Global Greedy Dependency Parsing},
	author={Li, Zuchao and Zhao, Hai and Parnow, Kevin},
  	booktitle={the Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-2020)},
	year={2020}
}
```