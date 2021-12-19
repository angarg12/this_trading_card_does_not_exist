#!/bin/bash
PYTHON_PATH=/home/tdimson/anaconda3/envs/company_makeup/bin/python
LIBRARY_PATH=/home/tdimson/projects/company-makeup/title_maker_pro

export PYTHONPATH=.:

python title_maker_pro/train.py  ^
--summary_comment=dict_words ^
--output_dir=models/dict_words ^
--model_type=gpt2 ^
--model_name_or_path=gpt2 ^
--do_train ^
--train_data_file=notebooks/dict_words.pickle ^
--do_eval ^
--eval_data_file=notebooks/dict_words.pickle ^
--per_gpu_train_batch_size 32 ^
--per_gpu_eval_batch_size 32 ^
--gradient_accumulation_steps 1 ^
--splits 0.95 --splits 0.05 ^
--train_split_idx 0 --eval_split_idx 1 ^
--evaluate_during_training ^
--save_steps 10000 ^
--logging_steps 2500 ^
--eval_subsampling 1.0 ^
--learning_rate 0.00001 ^
--block_size 512 ^
--num_train_epochs 1 ^
--overwrite_output_dir ^
--overwrite_cache