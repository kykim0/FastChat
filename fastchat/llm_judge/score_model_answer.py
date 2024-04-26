"""Compute rewards and selects the best-of-n model answers.

Usage:
python3 score_model_answer.py --model-path <path> \
    --model-id gemma-custom --num-gpus-total 4 \
    --answer-file data/mt_bench/model_answer/g2b-sft-256.jsonl \
    --bon-file data/mt_bench/model_answer/g2b-sft-16-btbinf-s42.jsonl \
    --best-of-n=64
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template


def run_rm(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    bon_file,
    score_file,
    best_of_n,
    num_gpus_total=1,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    answers = {}
    with open(answer_file, "r") as f:
        for l in f:
            answer = json.loads(l)
            answers[answer["question_id"]] = answer

    # Split the question file into `num_gpus` files
    use_ray = num_gpus_total > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=1)(
            get_model_scores
        ).remote
    else:
        get_answers_func = get_model_scores

    chunk_size = len(questions) // num_gpus_total
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answers,
                bon_file,
                score_file,
                best_of_n,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_scores(
    model_path,
    model_id,
    questions,
    answers,
    bon_file,
    score_file,
    best_of_n,
):
    rmodel_tokenizer = AutoTokenizer.from_pretrained(model_path)
    rmodel = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
    rmodel.eval()

    for question in tqdm(questions):
        answer_dict = answers[question["question_id"]]
        choice_rewards = []
        for choice in answer_dict["choices"][:best_of_n]:
            conv = get_conversation_template(model_id)
            prompts = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], choice["turns"][j])
                prompt = conv.get_prompt()
                prompts.append(prompt)

            with torch.no_grad():
                rmodel_inputs = rmodel_tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to("cuda")
                rmodel_outputs = rmodel(**rmodel_inputs).logits
            turn_rewards = [r.squeeze().cpu().item() for r in rmodel_outputs]
            choice_rewards.append({"index": choice["index"], "turns": turn_rewards})

        # Dump best-of-n answers.
        max_idx = np.argmax([np.mean(d["turns"]) for d in choice_rewards]).item()
        os.makedirs(os.path.dirname(bon_file), exist_ok=True)
        with open(os.path.expanduser(bon_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": answer_dict["answer_id"],
                "model_id": model_id,
                "choices": [answer_dict["choices"][max_idx]],
            }
            fout.write(json.dumps(ans_json) + "\n")

        # Dump raw scores.
        if not score_file: continue
        with open(os.path.expanduser(score_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": answer_dict["answer_id"],
                "model_id": model_id,
                "choices": choice_rewards,
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Reward model name or path.")
    parser.add_argument("--model-id", type=str, required=True, help="A custom name for the model.")
    parser.add_argument("--answer-file", type=str, required=True, help="Answer file path.")
    parser.add_argument("--bon-file", type=str, required=True, help="Output BoN file path.")
    parser.add_argument("--best-of-n", type=int, default=16, help="N for best-of-N.")
    parser.add_argument("--num-gpus-total", type=int, default=1, help="The total number of GPUs.")

    args = parser.parse_args()

    if args.num_gpus_total > 1:
        import ray

        ray.init()

    question_file = f"data/mt_bench/question.jsonl"
    answer_file = args.answer_file
    bon_file = args.bon_file
    if os.path.exists(bon_file):
        raise ValueError(f"{bon_file} already exists.")
    fname = os.path.basename(bon_file)
    score_file = None  # os.path.join(os.path.dirname(bon_file), f"{fname[:fname.find('.jsonl')]}_score.jsonl")
    run_rm(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=None,
        question_end=None,
        answer_file=args.answer_file,
        bon_file=bon_file,
        score_file=score_file,
        best_of_n=args.best_of_n,
        num_gpus_total=args.num_gpus_total,
    )

    reorg_answer_file(bon_file)
    if score_file:
        reorg_answer_file(score_file)
