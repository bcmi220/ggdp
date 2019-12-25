# coding=utf-8

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import re
import json

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from utils_dependency_parsing import convert_examples_to_features_with_parsing_order, get_labels, save_labels, read_examples_from_file, write_conll_examples
from tree_inference import decode_MST, decode_GGDP_projective, decode_GGDP_nonprojective


from configuration.configuration_bert import BertForDependencyParsingWithOrderConfig


from modeling.modeling_bert import BertForDependencyParsingWithOrder

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertForDependencyParsingWithOrderConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertForDependencyParsingWithOrderConfig, BertForDependencyParsingWithOrder, BertTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_data, eval_data, model, tokenizer, postags, labels):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    train_dataset = train_data[0]

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_results = {'las':0.0, 'uas':0.0}
    model.zero_grad()
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for global_epoch in range(int(args.num_train_epochs)):
        logger.info("***** Training epoch %d *****", global_epoch)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[1],
                      "attention_mask": batch[2],
                      "order_ids": batch[5] if args.use_postag else batch[4],
                      "head_ids": batch[6] if args.use_postag else batch[5],
                      "label_ids": batch[7] if args.use_postag else batch[6]
                    }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[3] if args.model_type in ["bert", "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
            
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and \
                    args.logging_steps == 1 and \
                    global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Eval on the dev dataset every X step
                if (args.local_rank == -1 or torch.distributed.get_rank() == 0) and \
                    args.evaluate_during_training and \
                    (args.eval_strategy == 1 or args.eval_strategy == 2) and \
                    global_step % args.eval_steps == 0:

                    results = evaluate(args, eval_data, model, tokenizer, labels)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    has_better_model = False
                    if results['uas'] > best_results['uas'] or (results['uas'] == best_results['uas'] and results['las'] > best_results['las']):
                        best_results = results
                        has_better_model = True
                        logger.info("New best results!")

                    if args.save_strategy == 0 or \
                        (args.save_strategy == 1 and has_better_model):
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-step-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        if args.use_postag:
                            save_labels(os.path.join(output_dir, "postags.txt"), postags)
                        save_labels(os.path.join(output_dir, "labels.txt"), labels)

                        with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
                            json.dump(results, f)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        
        # Eval on the dev dataset every X epoch
        if (args.local_rank == -1 or torch.distributed.get_rank() == 0) and \
            args.evaluate_during_training and \
            ((args.eval_strategy == 0 and global_epoch % args.eval_steps == 0) or args.eval_strategy == 2):

            results = evaluate(args, eval_data, model, tokenizer, labels)
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
            has_better_model = False
            if results['uas'] > best_results['uas'] or (results['uas'] == best_results['uas'] and results['las'] > best_results['las']):
                best_results = results
                has_better_model = True
                logger.info("New best results!")
            
            if args.save_strategy == 0 or \
                (args.save_strategy == 1 and has_better_model):
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint-epoch-{}".format(global_epoch))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                if args.use_postag:
                    save_labels(os.path.join(output_dir, "postags.txt"), postags)
                save_labels(os.path.join(output_dir, "labels.txt"), labels)

                with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
                    json.dump(results, f)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, eval_data, model, tokenizer, labels):

    eval_dataset, eval_examples, eval_features = eval_data

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    words = []
    postags = [] if args.use_postag else None
    gold_heads = []
    gold_labels = []
    pred_heads = []
    pred_labels = []
    model.eval()
    order_corr = 0
    order_sum = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[1],
                      "attention_mask": batch[2]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[3] if args.model_type in ["bert", "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)


            if args.infer_alg == 'ggp':
                # 
                _, energy_order = outputs[3].max(dim=2)
                energy = outputs[0] + energy_order.unsqueeze(1).unsqueeze(2)

                # [batch, num_labels, length, length]
                energy = energy.detach().cpu().numpy()
            else:
                # [batch, num_labels, length, length]
                energy = outputs[0].detach().cpu().numpy()

            # [batch, length, max_parsing_order]
            order_logits = outputs[3].detach().cpu().numpy()
            
            # [batch, length]
            batch_order_preds = np.argmax(order_logits, axis=2)

            example_ids = batch[0].cpu().numpy()
            
            # for every sentence
            for batch_i in range(energy.shape[0]):
                # [length]
                order_preds = batch_order_preds[batch_i]
                # [num_labels, length, length]
                energy_i = energy[batch_i]
                example_i = eval_examples[example_ids[batch_i]]
                feature_i = eval_features[example_ids[batch_i]]
                # [num_labels, word_length, word_length], <ROOT> in the first position
                energy_i = energy_i[:, feature_i.word_token_starts, :]
                energy_i = energy_i[:, :, feature_i.word_token_starts]

                # order accuracy
                order_preds = order_preds[feature_i.word_token_starts] 
                order_gold = np.array(example_i.orders)
                order_sum += order_gold.shape[0]
                order_corr += (order_preds[1:] == order_gold).sum() # remove root

                # decoding
                if args.infer_alg == 'mst':
                    head_preds, label_preds = decode_MST(energy_i, leading_symbolic=1, labeled=True)
                elif args.infer_alg == 'ggp': # global greedy projective
                    head_preds, label_preds = decode_GGDP_projective(energy_i, leading_symbolic=1, labeled=True)
                else: # global greedy non-projective
                    head_preds, label_preds = decode_GGDP_nonprojective(energy_i, order_preds.astype(float), leading_symbolic=1, labeled=True)

                # map labels
                label_preds = [labels[item] for item in label_preds]

                words.append(example_i.words)
                if args.use_postag:
                    postags.append(example_i.postags)
                gold_heads.append(example_i.heads)
                gold_labels.append(example_i.labels)
                pred_heads.append(head_preds)
                pred_labels.append(label_preds)
    
    # write reference file
    gold_output_file = os.path.join(args.output_dir, 'eval.gold')
    write_conll_examples(words, postags, gold_heads, gold_labels, gold_output_file)
    # write predict file
    eval_output_file = os.path.join(args.output_dir, 'eval.pred')
    write_conll_examples(words, postags, pred_heads, pred_labels, eval_output_file)
    # eval
    eval_f = os.popen("perl "+os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.pl")+" -q -g "+gold_output_file+" -s "+eval_output_file, "r")
    result_text = eval_f.read().strip()
    logger.info("***** Eval info *****")
    logger.info(result_text)
    eval_f.close()

    eval_las = re.findall(r'Labeled attachment score: \d+ / \d+ \* \d+ = ([\d\.]+) \%', result_text)
    if len(eval_las) > 0:
        eval_las = float(eval_las[0])
    else:
        eval_las = 0.0
    
    eval_uas = re.findall(r'Unlabeled attachment score: \d+ / \d+ \* \d+ = ([\d\.]+) \%', result_text)
    if len(eval_uas) > 0:
        eval_uas = float(eval_uas[0])
    else:
        eval_uas = 0.0

    results = {
        "uas": eval_uas,
        "las": eval_las,
        "order_acc": float('%.2f' % (order_corr / order_sum * 100))
    }
    
    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("eval_%s = %s", key, str(results[key]))
    
    return results



def predict(args, predict_data, model, tokenizer, labels):

    predict_dataset, predict_examples, predict_features = predict_data

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    predict_sampler = SequentialSampler(predict_dataset) if args.local_rank == -1 else DistributedSampler(predict_dataset)
    predict_dataloader = DataLoader(predict_dataset, sampler=predict_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Eval!
    logger.info("***** Running prediction *****")
    logger.info("  Num examples = %d", len(predict_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    words = []
    postags = [] if args.use_postag else None
    pred_heads = []
    pred_labels = []
    model.eval()

    for batch in tqdm(predict_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[1],
                      "attention_mask": batch[2]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[3] if args.model_type in ["bert", "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)


            if args.infer_alg == 'ggp':
                # 
                _, energy_order = outputs[3].max(dim=2)
                energy = outputs[0] + energy_order.unsqueeze(1).unsqueeze(2)

                # [batch, num_labels, length, length]
                energy = energy.detach().cpu().numpy()
            else:
                # [batch, num_labels, length, length]
                energy = outputs[0].detach().cpu().numpy()

            # [batch, length, max_parsing_order]
            order_logits = outputs[3].detach().cpu().numpy()
            
            # [batch, length]
            batch_order_preds = np.argmax(order_logits, axis=2)

            example_ids = batch[0].cpu().numpy()
            
            # for every sentence
            for batch_i in range(energy.shape[0]):
                # [length]
                order_preds = batch_order_preds[batch_i]
                # [num_labels, length, length]
                energy_i = energy[batch_i]
                example_i = predict_examples[example_ids[batch_i]]
                feature_i = predict_features[example_ids[batch_i]]
                # [num_labels, word_length, word_length], <ROOT> in the first position
                energy_i = energy_i[:, feature_i.word_token_starts, :]
                energy_i = energy_i[:, :, feature_i.word_token_starts]

                # order accuracy
                order_preds = order_preds[feature_i.word_token_starts] 

                # decoding
                if args.infer_alg == 'mst':
                    head_preds, label_preds = decode_MST(energy_i, leading_symbolic=1, labeled=True)
                elif args.infer_alg == 'ggp': # global greedy projective
                    head_preds, label_preds = decode_GGDP_projective(energy_i, leading_symbolic=1, labeled=True)
                else: # global greedy non-projective
                    head_preds, label_preds = decode_GGDP_nonprojective(energy_i, order_preds.astype(float), leading_symbolic=1, labeled=True)

                # map labels
                label_preds = [labels[item] for item in label_preds]

                words.append(example_i.words)
                if args.use_postag:
                    postags.append(example_i.postags)
                pred_heads.append(head_preds)
                pred_labels.append(label_preds)
    
    # write predict file
    predict_output_file = os.path.join(args.output_dir, args.predict_output)
    write_conll_examples(words, postags, pred_heads, pred_labels, predict_output_file)
    

def load_examples(args, file_name, tokenizer, is_training=False, postags=None, labels=None, pad_postag='_', pad_label='_', convert_strategy=0, special_postag='_', special_label='_'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset

    logger.info("Creating features from dataset file: %s", os.path.join(args.data_dir, file_name))

    examples = read_examples_from_file(args.data_dir, file_name, is_training=is_training, use_postag=args.use_postag, with_parsing_order=True)

    features = convert_examples_to_features_with_parsing_order(
        examples, args.max_seq_length, tokenizer,
        cls_token_at_end=bool(args.model_type in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(args.model_type in ["roberta"]),
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(args.model_type in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        is_training=is_training,
        max_parsing_order=args.max_parsing_order,
        use_postag=args.use_postag,
        postag_list=postags,
        label_list=labels,
        pad_postag=pad_postag,
        pad_label=pad_label,
        convert_strategy=convert_strategy,
        special_postag=special_postag,
        special_label=special_label
    )

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset

    # Convert to Tensors and build dataset
    all_example_ids = torch.tensor([f.index for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if args.use_postag:
        all_postag_ids = torch.tensor([f.postag_ids for f in features], dtype=torch.long)
    if is_training:
        all_order_ids = torch.tensor([f.order_ids for f in features], dtype=torch.long)
        all_head_ids = torch.tensor([f.head_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    
    if args.use_postag and is_training:
        dataset = TensorDataset(all_example_ids, all_input_ids, all_input_mask, all_segment_ids, all_postag_ids, all_order_ids, all_head_ids, all_label_ids)
    elif args.use_postag:
        dataset = TensorDataset(all_example_ids, all_input_ids, all_input_mask, all_segment_ids, all_postag_ids)
    elif is_training:
        dataset = TensorDataset(all_example_ids, all_input_ids, all_input_mask, all_segment_ids, all_order_ids, all_head_ids, all_label_ids)
    else:
        dataset = TensorDataset(all_example_ids, all_input_ids, all_input_mask, all_segment_ids)
    
    return (dataset, examples, features)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the files for the Denpendency Parsing task.")
    parser.add_argument("--train_file", default=None, type=str,
                        help="The train file name for the Denpendency Parsing task.")
    parser.add_argument("--eval_file", default=None, type=str,
                        help="The eval file name for the Denpendency Parsing task.")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="The eval file name for the Denpendency Parsing task.")
    parser.add_argument("--predict_output", default=None, type=str,
                        help="The predict output file name for the Denpendency Parsing task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--arc_space", default=512, type=int,
                        help="Arc hidden size.")
    parser.add_argument("--label_space", default=128, type=int,
                        help="Label hidden size.")
    parser.add_argument("--order_space", default=128, type=int,
                        help="Order hidden size.")

    parser.add_argument("--max_parsing_order", default=32, type=int,
                        help="Maximum parsing order.")

    parser.add_argument("--infer_alg", default='mst', type=str,
                        help="Model type selected in the list: " + ", ".join(['mst', 'ggp', 'ggnp']))

    ## Other parameters
    parser.add_argument("--use_postag", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--postags", default="", type=str,
                        help="Path to a file containing all postag labels.")
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all dependency labels.")
    parser.add_argument("--convert_strategy", default=0, type=int,
                        help="Word-level to subword-level dependency tree convert strategy.")
    parser.add_argument("--special_postag", default="_", type=str,
                        help="Special postag for the subword due to word-level to subword-level conversion.")
    parser.add_argument("--special_label", default="_", type=str,
                        help="Special label for the subword due to word-level to subword-level conversion.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--eval_strategy", type=int, default=0,
                        help="Eval model strategy. 0 for every X epochs, 1 for every X steps, 2 for every X steps and every epoch.")
    parser.add_argument("--eval_steps", type=int, default=1,
                        help="Eval model every X updates steps.")
    parser.add_argument("--save_strategy", type=int, default=0,
                        help="Save checkpoint strategy. 0 for save all checkpoints, 1 for save better checkpoints.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    if not os.path.exists(args.output_dir) and args.do_train:
        os.makedirs(args.output_dir)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare dependency parsing task
    postags = None
    if args.use_postag:
        if len(args.postags) > 0:
            postags = get_labels(args.postags)
        else:
            postags = get_labels(os.path.join(args.model_name_or_path, 'postags.txt'))
        assert postags[0] == '_', 'PAD postag must be in the position 0'
    
    if len(args.labels) > 0:
        labels = get_labels(args.labels)
    else:
        labels = get_labels(os.path.join(args.model_name_or_path, 'labels.txt'))
    assert labels[0] == '_', 'PAD label must be in the position 0'
    if args.convert_strategy > 0:
        if args.use_postag:
            assert args.special_postag in postags
        assert args.special_label in labels
    
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.do_train:
        config = config_class.from_pretrained(
                                    args.config_name if args.config_name else args.model_name_or_path,
                                    cache_dir=args.cache_dir if args.cache_dir else None,
                                    use_postag=args.use_postag, num_postags=len(postags) if postags is not None else 0,
                                    num_labels=len(labels), 
                                    arc_space=args.arc_space, label_space=args.label_space, 
                                    max_parsing_order=args.max_parsing_order,
                                    order_space=args.order_space
                                )
    else:
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                            cache_dir=args.cache_dir if args.cache_dir else None,)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        assert args.train_file
        train_data = load_examples(
                            args, args.train_file, tokenizer, 
                            is_training=True,
                            postags=postags, 
                            labels=labels, 
                            pad_postag='_',
                            pad_label='_',
                            convert_strategy=args.convert_strategy,
                            special_postag=args.special_postag,
                            special_label=args.special_label
                        )
    
    if args.do_eval:
        assert args.eval_file
        eval_data = load_examples(
                            args, args.eval_file, tokenizer, 
                            is_training=True,
                            postags=postags, 
                            labels=labels, 
                            pad_postag='_',
                            pad_label='_',
                            convert_strategy=args.convert_strategy,
                            special_postag=args.special_postag,
                            special_label=args.special_label
                        )
    
    if args.do_train:

        global_step, tr_loss = train(args, train_data, eval_data, model, tokenizer, postags, labels)

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.local_rank in [-1, 0]:
            # Create output directory if needed
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            if args.use_postag:
                save_labels(os.path.join(args.output_dir, "postags.txt"), postags)
            save_labels(os.path.join(args.output_dir, "labels.txt"), labels)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    if args.do_eval and args.local_rank in [-1, 0]:
        evaluate(args, eval_data, model, tokenizer, labels)

    if args.do_predict and args.local_rank in [-1, 0]:
        assert args.predict_file and args.predict_output
        predict_data = load_examples(
                            args, args.predict_file, tokenizer, 
                            is_training=False,
                            postags=postags, 
                            labels=labels, 
                            pad_postag='_',
                            pad_label='_',
                            convert_strategy=args.convert_strategy,
                            special_postag=args.special_postag,
                            special_label=args.special_label
                        )
        predict(args, predict_data, model, tokenizer, labels)


if __name__ == "__main__":
    main()