# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import argparse
from functools import partial
from tqdm import tqdm
import wandb
import pickle
import dill

# print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")

# HF classses
from transformers import LogitsProcessorList, DataCollatorWithPadding

# better bool flag type for argparse
from utils.submitit import str2bool

# some file i/o helpers
from utils.io import write_jsonlines, write_json

# watermarking functionality
from watermark_processor import WatermarkLogitsProcessor

import watermarks


# generation pipeline helpers
from utils.generation import (
    MAX_GENERATIONS,
    load_model,
    load_hf_dataset,
    check_input_lengths,
    check_output_lengths,
    tokenize_for_generation,
    generate,
)


def main(args):
    ###########################################################################
    # Start logging
    ###########################################################################
    # storing slurm info to allow auditing logfiles later
    args.SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
    args.SLURM_ARRAY_JOB_ID = os.getenv("SLURM_ARRAY_JOB_ID")
    args.SLURM_ARRAY_TASK_ID = os.getenv("SLURM_ARRAY_TASK_ID")

    if args.wandb:
        # start a new wandb run to track this experiment, will send data to it later
        run = wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.run_name}",
            # track hyperparameters and run metadata
            config=args,
            tags=args.wandb_tags,
        )

    ###########################################################################
    # Create the output dir
    ###########################################################################
    print(f"Output dir for this run: {args.output_dir}")
    # notify if exists
    if os.path.exists(args.output_dir):
        print(f"Output dir for this run already exists!")
        print(f"Contents: {sorted(os.listdir(args.output_dir))}")
    else:
        # create the output dir where run artifacts are stored
        os.makedirs(args.output_dir)

    ###########################################################################
    # Load the dataset
    ###########################################################################
    # basic ops like shuffling and select are done in load fn
    dataset = load_hf_dataset(args)

    ###########################################################################
    # Instantiate model and tokenizer
    ###########################################################################

    model, tokenizer, device = load_model(args)

    ###########################################################################
    # Configure the prompt construction partial
    ###########################################################################

    # Construct the data filtering/sampling scheme partials
    token_kwargs = dict(
        hf_model_name=args.model_name_or_path,
        tokenizer=tokenizer,
        args=args,
    )
    if args.input_truncation_strategy == "prompt_length":
        token_kwargs.update(dict(min_prompt_tokens=args.min_prompt_tokens))
    elif args.input_truncation_strategy == "completion_length":
        token_kwargs.update(dict(max_new_tokens=args.max_new_tokens))
    elif args.input_truncation_strategy == "no_truncation":
        # truncate_input_for_prompt is a bool flag, that is set by
        # the dataset loading function, semi-redundant, to make sure
        # people are very aware of which input data style they are using
        assert (
            args.truncate_input_for_prompt == False
        ), "Cannot truncate input for prompt if 'no_truncation' strategy is specified"
        pass
    else:
        ValueError(f"Unknown input truncation strategy {args.input_truncation_strategy}")
    tokenize_prompts = partial(tokenize_for_generation, **token_kwargs)

    ###########################################################################
    # Configure the I/O data validation partials
    ###########################################################################

    input_check_kwargs = dict(
        min_sample_len=args.min_sample_tokens,
        max_input_len=model.config.max_position_embeddings,
        max_new_tokens=args.max_new_tokens,
    )
    if args.input_filtering_strategy == "prompt_length":
        input_check_kwargs.update(dict(min_prompt_len=args.min_prompt_tokens, min_completion_len=0))
    elif args.input_filtering_strategy == "completion_length":
        input_check_kwargs.update(dict(min_prompt_len=0, min_completion_len=args.max_new_tokens))
    elif args.input_filtering_strategy == "prompt_and_completion_length":
        input_check_kwargs.update(
            dict(min_prompt_len=args.min_prompt_tokens, min_completion_len=args.max_new_tokens)
        )
    elif args.input_filtering_strategy == "no_filter":
        input_check_kwargs.update(dict(min_prompt_len=0, min_completion_len=0))
    else:
        ValueError(f"Unknown input filtering strategy {args.input_filtering_strategy}")
    input_check = partial(check_input_lengths, **input_check_kwargs)

    if args.output_filtering_strategy == "max_new_tokens":
        output_kwargs = dict(min_output_len=args.max_new_tokens)
    elif args.output_filtering_strategy == "no_filter":
        output_kwargs = dict(min_output_len=0)
    else:
        ValueError(f"Unknown output filtering strategy {args.output_filtering_strategy}")
    output_check = partial(check_output_lengths, **output_kwargs)

    ###########################################################################
    # Construct the watermark processor
    ###########################################################################

    match args.watermark:
        case 'john23':
            watermark_processor = watermarks.john23_WatermarkLogitsProcessor(
                                                    vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    seeding_scheme=args.seeding_scheme,
                                                    # store_spike_ents=args.store_spike_ents,
                                                    select_green_tokens=True)
            
        case 'xuandong23b':
            watermark_processor = watermarks.xuandong23b_WatermarkLogitsProcessor(fraction=args.fraction,
                                                    strength=args.strength,
                                                    vocab_size=model.config.vocab_size,
                                                    watermark_key=args.wm_key)
            
        case 'rohith23':
            watermark_processor = watermarks.rohith23_WatermarkLogitsProcessor(vocab_size=model.config.vocab_size)
        case 'lean23':
            watermark_processor = watermarks.lean23_BalanceMarkingWatermarkLogitsProcessor(tokenizer=tokenizer,
                                                                        lm_tokenizer=tokenizer,
                                                                        lm_model=model,
                                                                        delta=args.lean_delta,
                                                                        lm_prefix_len=args.lm_prefix_len,
                                                                        lm_top_k=args.lm_top_k,
                                                                        message_code_len=args.message_code_len,
                                                                        random_permutation_num=args.random_permutation_num,
                                                                        encode_ratio=args.encode_ratio,
                                                                        max_confidence_lbd=args.max_confidence_lbd,
                                                                        message_model_strategy=args.message_model_strategy,
                                                                        message=args.message,
                                                                        top_k=args.lean23_top_k,
                                                                        repeat_penalty=args.repeat_penalty
                                                                        )
            dill.dump(watermark_processor.watermark_processor, open("~/sok-llm-watermark/watermark_reliability_release/watermarks/lean23/processor/lean23_"+str(args.model_name_or_path).replace("/","")+".pkl", "wb"))
            '''filename = "~/codable-watermarking-for-llm/gen_table.jsonl"
            with open(filename, "r", encoding="utf-8") as f:
                c4_sliced_and_filted = [json.loads(line) for line in f.read().strip().split("\n")]
                decoded_message_list = []
                other_information_list = []
                for text in c4_sliced_and_filted:
                    tokenized_input = tokenizer(text['truncated_input'], return_tensors='pt').to(model.device)
                    #tokenized_input = truncate(tokenized_input, max_length=args.prompt_length)

                    ### Could be problem here?
                    temperature = 1.0
                    generated_length=200
                    watermark_processor.logit_processor[2].start_length = tokenized_input['input_ids'].shape[-1]
                    output_tokens = model.generate(**tokenized_input,
                                                temperature=temperature,
                                                max_new_tokens=generated_length,
                                                num_beams=args.num_beams,
                                                logits_processor=[watermark_processor])

                    output_text = \
                        tokenizer.batch_decode(
                            output_tokens[:, tokenized_input["input_ids"].shape[-1]:],
                            skip_special_tokens=True)[0]


                    decoded_message, other_information = watermark_processor.logit_processor[2].decode(output_text, disable_tqdm=True)
                    decoded_message_list.append(decoded_message)
                    other_information_list.append(other_information)
                    print()
    
            print()'''
            ### Delete this line
            '''args_list = [args.lean_delta, 
                         args.lm_prefix_len, 
                         args.lm_top_k, 
                         args.message_code_len, 
                         args.random_permutation_num, 
                         args.encode_ratio, 
                         args.max_confidence_lbd, 
                         args.message_model_strategy, 
                         args.message, 
                         args.lean23_top_k, 
                         args.repeat_penalty]
            dill.dump(args_list, open("~/sok-llm-watermark/watermark_reliability_release/watermarks/lean23/processor/lean23_args.pkl", "wb"))
            dill.dump(watermark_processor, open("~/sok-llm-watermark/watermark_reliability_release/watermarks/lean23/processor/lean23.pkl", "wb"))'''


        case 'aiwei23':
            data_dir = args.data_dir+str(args.model_name_or_path)+"/"
            model_dir = args.model_dir+str(args.model_name_or_path)+"/"
            if args.use_sampling:
                data_dir += str(args.sampling_temp) + "/"
                model_dir += str(args.sampling_temp) + "/"
            if args.aiwei_trained:
                watermark_detector = watermarks.aiwei23_WatermarkDetector(bit_number=args.bit_number,
                                                        window_size=args.window_size,
                                                        layers=args.layers,
                                                        gamma=args.gamma,
                                                        delta=args.delta,
                                                        lm_model = model,
                                                        lm_tokenizer = tokenizer,
                                                        beam_size=args.num_beams,
                                                        data_dir=data_dir,
                                                        z_value=args.z_value,
                                                        model_dir=model_dir,
                                                        llm_name = args.model_name_or_path)
                watermark_detector.get_detector_model()
                watermark_processor = watermark_detector.build_logits_processor()
                print("Load processor and detector done.")
            else:
                watermarks.prepare_generator(bit_number=args.bit_number,
                                    layers=args.layers,
                                    sample_number=args.sample_number,
                                    window_size=args.window_size,
                                    data_dir=data_dir+"train_generator_data/train_generation_data.jsonl",
                                    model_dir=model_dir)
                
                watermark_detector = watermarks.aiwei23_WatermarkDetector(bit_number=args.bit_number,
                                                        window_size=args.window_size,
                                                        layers=args.layers,
                                                        gamma=args.gamma,
                                                        delta=args.delta,
                                                        llm_name = args.model_name_or_path,
                                                        beam_size=args.num_beams,
                                                        data_dir=data_dir,
                                                        z_value=args.z_value,
                                                        model_dir=model_dir)

                watermark_detector.generate_and_save_train_data(num_samples=args.num_samples)
                watermark_processor = watermark_detector.generate_and_save_test_data(dataset_name=args.train_dataset_name,
                                                                            sampling_temp=args.sampling_temp,
                                                                            max_new_tokens=args.max_new_tokens,args=args)
                
                watermark_detector.train_model(output_model_dir=model_dir)
                print()


        case 'kiyoon23':
            watermark_processor = watermarks.kiyoon23(args.dtype, args.embed, args.exp_name_generic, args.exp_name_infill,
                                              args.extract,
                                              args.num_sample, args.spacy_model, args.exclude_cc, args.custom_keywords,
                                              args.keyword_mask,
                                              args.keyword_ratio, args.mask_order_by, args.mask_select_method,
                                              args.num_epochs,
                                              args.topk, args.message)


        case 'xiaoniu23':
            if args.watermark_type == 'delta':
                reweight = watermarks.Delta_Reweight()
            elif args.watermark_type == 'gamma':
                reweight = watermarks.Gamma_Reweight()
            
            watermark_processor = watermarks.xiaoniu23_WatermarkLogitsProcessor(
                args.private_key.encode("utf-8"),
                reweight,
                watermarks.PrevN_ContextCodeExtractor(args.n))
            
            watermarks.patch_model(model)

        case 'aiwei23b':
            if not args.aiwei23b_trained:
                watermarks.aiwei23b_prepare_watermark_model(args.embedding_input_path, 
                                                            args.embedding_output_path, 
                                                            args.aiwei23b_model_path, 
                                                            args.aiwei23b_size, 
                                                            args.aiwei23b_output_model, 
                                                            args.watermark_model_epochs, 
                                                            args.watermark_model_lr, 
                                                            args.aiwei23b_input_dim, 
                                                            args.mapping_length, 
                                                            args.mapping_output_dir)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id

            if args.aiwei23b_watermark_type == "window": # use a window of previous tokens to hash, e.g. KGW
                watermark_model = watermarks.aiwei23b_watermark.WatermarkWindow(device, args.aiwei23b_window_size, tokenizer)
                watermark_processor = WatermarkLogitsProcessor(watermark_model)
            elif args.aiwei23b_watermark_type == "context":
                watermark_model = watermarks.aiwei23b_watermark.WatermarkContext(device, args.aiwei23b_chunk_size, tokenizer, delta = args.aiwei23b_delta,transform_model_path=args.transform_model, embedding_model=args.embedding_model)
                watermark_processor = watermarks.aiwei23b_WatermarkLogitsProcessor(watermark_model)

        # case "christ23":
                                                    
        #     watermark_processor = watermarks.christ23_WatermarkLogitsProcessor(tokenizer=tokenizer,vocab_size=model.config.vocab_size, temp=args.sampling_temp,  device=device)
            
        ######################################################################
        # Add your code here
        ######################################################################
        # If you have new watermark, add them here
        ######################################################################   
    
        case _:
            raise ValueError(f"Unknown watermark type: {args.watermark}")

    ###########################################################################
    # Configure the generation partials
    ###########################################################################

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    # FIXME can add typica
    if args.use_sampling:
        gen_kwargs.update(
            dict(
                do_sample=True,
                top_k=args.top_k,
                top_p=args.top_p,
                typical_p=args.typical_p,
                temperature=args.sampling_temp,
            )
        )
    else:
        gen_kwargs.update(dict(num_beams=args.num_beams))

    if args.watermark == 'kiyoon23':
        generate_with_watermark = watermark_processor.embed_watermark
        generate_without_watermark = partial(model.generate, **gen_kwargs)
    elif args.watermark == 'xiaoniu23':
        generate_with_watermark = partial(
            watermarks.generate_with_watermark_xiaoniu23, model_str=args.model_name_or_path,
            wp=[watermark_processor], 
            **gen_kwargs,
        )
        generate_without_watermark = partial(model.generate, **gen_kwargs)
    else:
        generate_without_watermark = partial(model.generate, **gen_kwargs)
        generate_with_watermark = partial(
            model.generate, logits_processor=LogitsProcessorList([watermark_processor]), **gen_kwargs
        )

    # construct the collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)

    generation_partial = partial(
        generate,
        data_collator=data_collator,
        generate_without_watermark=generate_without_watermark,
        generate_with_watermark=generate_with_watermark,
        watermark_processor=watermark_processor,
        tokenizer=tokenizer,
        device=device,
        args=args,
    )

    ###########################################################################
    # Compose the partials to create the pipeline
    ###########################################################################

    # tokenize and truncate the row inputs to create prompts according to the strategy spec'd above
    dataset_w_prompts = dataset.map(tokenize_prompts, batched=False)

    # filter the rows of the dataset based on length checks for the tokenized prompts and baseline completions
    dataset_input_len_filtered = dataset_w_prompts.filter(input_check, batched=False)

    # need to remove the input tensor column after this map
    # bc it persists between the prompt creation and generation maps
    columns_to_remove = args.columns_to_remove + ["input_ids"]

    # call the generation partial on each prompt in the dataset
    dataset_w_generations = dataset_input_len_filtered.map(
        generation_partial,
        batched=True,
        batch_size=args.generation_batch_size,
        remove_columns=columns_to_remove,
    )

    ###########################################################################
    # Main loop - actually executes the generation pipeline.
    # and accumulates the result rows in a list, assumes list is "small"-ish
    # and we aren't accumulating any tensors or other memory hogging artifacts
    ###########################################################################

    processed_examples = []
    ds_iterator = iter(dataset_w_generations)
    i = 0
    total_steps = 0
    pbar = tqdm(total=args.min_generations)
    while i < args.min_generations:
        try:
            ex = next(ds_iterator)
            total_steps += 1
        except StopIteration:
            break

        if args.verbose:
            # log basics to stdout
            print(f"#" * 80)
            print(f"dataset index: {ex['idx']}")
            print(f"orig_sample_length: {ex['orig_sample_length']}")
            print(f"prompt_length: {ex['prompt_length']}")
            print(f"real_completion_length: {ex['baseline_completion_length']}")
            print(f"no_wm_output_length: {ex['no_wm_output_length']}")
            print(f"w_wm_output_length: {ex['w_wm_output_length']}")

            print(f"\ntruncated_input: ")
            print(ex["truncated_input"])
            print(f"\nbaseline_completion: ")
            print(ex["baseline_completion"])
            print(f"\nno_wm_output: ")
            print(ex["no_wm_output"])
            print(f"\nw_wm_output: ")
            print(ex["w_wm_output"])

        processed_examples.append(ex)

        if output_check(ex):
            i += 1
            pbar.update(1)
        else:
            print(
                f"\n{i} of {len(processed_examples)} rows were satisfactory so far, {round(i/args.min_generations, 2)} of total.",
                f"\nCurrent generation overhead ratio: {round(len(processed_examples)/(i+1), 3)}.",
            )
        # if using wandb, log progress to wandb
        if args.wandb:
            run.log(
                {
                    "num_satisfactory_samples": i,
                    "progress_ratio": i / args.min_generations,
                    "generation_overhead_ratio": len(processed_examples) / (i + 1),
                    "total_generated_samples": len(processed_examples),
                },
                step=total_steps,
            )
    pbar.close()

    print(
        f"#" * 80,
        f"\nGeneration output length check overhead was num rows processed={len(processed_examples)}",
        f"for {args.min_generations} samples. Ratio: {round(len(processed_examples)/args.min_generations, 3)}",
    )
    if i < args.min_generations:
        print(
            f"#" * 80,
            f"\nWarning, may have run out of data before {args.min_generations} satisfactory samples were generated. ",
            f"\nNote, raw dataset limit was {args.limit_indices} rows.",
            f"\n{len(processed_examples)} prompt passed input checks and yielded generations, and {i} passed output checks,",
            f"\nProgress made: {round(i/args.min_generations, 2)}",
        )

    ###########################################################################
    # Generation jsonl dumping
    ###########################################################################

    gen_table_meta_path = f"{args.output_dir}/gen_table_meta.json"
    gen_table_path = f"{args.output_dir}/gen_table.jsonl"
    safe_gen_table_path = f"{args.output_dir}/gen_table_safe.jsonl"

    args.gen_table_already_existed = False

    if os.path.exists(gen_table_path):
        args.gen_table_already_existed = True
        print(f"Found existing generation files at this output dir: {args.output_dir}")
        if args.overwrite:
            print("Overwriting old generation files.")
            gen_table_path = gen_table_path
        else:
            print(
                f"Writing generations at alternate, safe path and exiting. Note! this only works once. "
                f"Safe version will get overwritten next time ... "
            )
            gen_table_path = safe_gen_table_path

    gen_table_meta = args.__dict__
    gen_table = processed_examples

    write_jsonlines(gen_table, gen_table_path)
    write_json(gen_table_meta, gen_table_meta_path, indent=4)

    # finish the wandb run
    if args.wandb:
        run.finish()
    return  # reload in separate script for metric measurement


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run watermarked huggingface LM generation pipeline"
    )
    parser.add_argument(
        "--watermark",
        type=str,
        default='',
        help="Select the watermark type",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-1.3b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=True,
        help="Whether to run model in float16 precsion.",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="c4",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="realnewslike",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="The split of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--stream_dataset",
        type=str2bool,
        default=True,
        help="Whether to stream the dataset from the web or download it locally.",
    )
    parser.add_argument(
        "--columns_to_remove",
        type=str,
        default=None,
        help="Comma separated list of columns to remove from the dataset before generation.",
    )
    parser.add_argument(
        "--shuffle_dataset",
        type=str2bool,
        default=False,
        help="Whether to shuffle the dataset before sampling.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=1234,
        help="The seed to use for dataset shuffle op.",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10_000,
        help="The buffer size to use for dataset shuffle op - takes n rows first, then shuffles those indices",
    )
    parser.add_argument(
        "--prompt_id",
        type=int,
        default=0,
        help="If the dataset supports multiple instruction prompts, denotes which one to use. 0 is default/no prompt.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="The number of tokens to generate using the model, and the num tokens removed from real text sample",
    )
    parser.add_argument(
        "--min_prompt_tokens",
        type=int,
        default=50,  # 500
        help="The number of examples (first N) to process from the dataset.",
    )
    parser.add_argument(
        "--min_sample_tokens",
        type=int,
        default=0,
        help="The the minimum length of raw prompt samples to consider.",
    )
    parser.add_argument(
        "--limit_indices",
        type=int,
        default=None,
        help="The number of examples (first N) to pull from the dataset, if None, pull all, and then set this arg to the number of rows in the dataset.",
    )
    parser.add_argument(
        "--min_generations",
        type=int,
        default=500,
        help="The minimum number of valid generations according to the output check strat to sample.",
    )
    parser.add_argument(
        "--input_truncation_strategy",
        type=str,
        default="completion_length",
        choices=["no_truncation", "completion_length", "prompt_length"],
        help="The strategy to use when tokenizing and truncating raw inputs to make prompts.",
    )
    parser.add_argument(
        "--input_filtering_strategy",
        type=str,
        default="completion_length",
        choices=["no_filter", "completion_length", "prompt_length", "prompt_and_completion_length"],
        help="The strategy to use when tokenizing and truncating raw inputs to make prompts.",
    )
    parser.add_argument(
        "--output_filtering_strategy",
        type=str,
        default="no_filter",
        choices=["no_filter", "max_new_tokens"],
        help=(
            f"The strategy to use when filtering/skipping rows if the model didn't ",
            f"generate enough tokens to facilitate analysis.",
        ),
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=False,
        help=("Whether to perform sampling during generation. (non-greedy decoding)"),
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="The temperature to use when generating using multinom sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="The top k to use when generating using top_k version of multinom sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="The top p to use when generating using top_p version of sampling",
    )
    parser.add_argument(
        "--typical_p",
        type=float,
        default=1.0,
        help="The typical p to use when generating using typical decoding version of multinom sampling",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams to use where '1' is no beam search.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=None,
        help="Seed for setting the torch rng prior to generation using any decoding scheme with randomness.",
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=4,
        help="The batch size to use for generation.",
    )
    
    

    
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="Whether to log the generations to stdout.",
    )
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=True,
        help="Whether to log to wandb.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="test-gen-eva",
        help="The name of the wandb project.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="alps-lab-sok",
        help="The wandb entity/user for the project.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="The comma separated list of tags to add to the wandb run.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="The unique name for the run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The unique name for the run.",
    )
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=False,
        help="Allow overwriting of old generation files at the same output location.",
    )
    args = parser.parse_args()
    
    match args.watermark:
        case 'john23':
            parser.add_argument(
                "--seeding_scheme",
                type=str,
                default="simple_1",
                help="The seeding procedure to use for the watermark.",
            )
            parser.add_argument(
                "--gamma",
                type=float,
                default=0.25,
                help="The ratio of tokens to put in the greenlist when splitting the vocabulary",
            )
            parser.add_argument(
                "--delta",
                type=float,
                default=2.0,
                help="The amount of bias (absolute) to add to the logits in the whitelist half of the vocabulary at every step",
            )
            parser.add_argument(
                "--store_spike_ents",
                type=str2bool,
                default=True,
                help=("Whether to store the spike entropies while generating with watermark processor. "),
            )    
        case 'xuandong23b':    

            parser.add_argument("--fraction", type=float, default=0.5)
            parser.add_argument("--strength", type=float, default=2.0)
            parser.add_argument("--wm_key", type=int, default=0)
        case 'lean23':    
            parser.add_argument("--lean_delta", type=float, default=1.5)
            parser.add_argument("--lm_prefix_len", type=int, default=10)
            parser.add_argument("--lm_top_k", type=int, default=-1),
            parser.add_argument("--message_code_len", type=int, default=20)
            parser.add_argument("--random_permutation_num", type=int, default=100)
            parser.add_argument("--encode_ratio", type=float, default=10.0)
            parser.add_argument("--max_confidence_lbd", type=float, default=0.5)
            parser.add_argument("--message_model_strategy", type=str, default="vanilla")
            parser.add_argument("--message", type=list, default=[100,200,300,400,500])
            parser.add_argument("--lean23_top_k", type=int, default=1000)
            parser.add_argument("--repeat_penalty", type=float, default=1.5)
            # parser.add_argument("--generated_length", type=int, default=200)
            parser.add_argument("--prompt_length", type=int, default=300)
        case 'aiwei23':    
            parser.add_argument("--bit_number", type=int, default=16) ### This is log2(vocab_size), which depends on the model, for opt, it is 16
            parser.add_argument("--layers", type=int, default=9)
            parser.add_argument("--window_size", type=int, default=3)
            #parser.add_argument("--llm_name", type=str, default="facebook/opt-1.3b")
            parser.add_argument("--gamma", type=float, default=0.5)
            parser.add_argument("--delta", type=float, default= 2.0)
            parser.add_argument("--model_dir", type=str, default="~/sok-llm-watermark/watermark_reliability_release/watermarks/aiwei23/model/")
            # parser.add_argument("--beam_size", type=int, default=0)
            parser.add_argument("--data_dir", type=str, default="~/sok-llm-watermark/watermark_reliability_release/watermarks/aiwei23/data/")
            parser.add_argument("--z_value", type=int, default=1)
            parser.add_argument("--sample_number", type=int, default=2000, help="Number of samples for training generator.")
            parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples for training detector.")
            parser.add_argument("--train_dataset_name", type=str, default="c4", help="The dataset used for training detector.")
            # parser.add_argument("--sampling_temp", type=float, default=0.7)
            # parser.add_argument("--max_new_token", type=int, default=100)
            parser.add_argument("--aiwei_trained", type=str2bool, default="True")
        case 'kiyoon23':
            parser.add_argument("--exp_name_generic", type=str, default="tmp")
            parser.add_argument("--embed", type=str2bool, default=False)
            parser.add_argument("--extract", type=str2bool, default=False)
            parser.add_argument("--dtype", type=str, default="agnews")
            parser.add_argument("--num_sample", type=int, default=200)
            parser.add_argument("--exp_name_infill", type=str, default="")
            parser.add_argument("--num_epochs", type=int, default=10)
            parser.add_argument("--keyword_ratio", type=float, default=0.05)
            parser.add_argument("--topk", type=int, default=2)
            parser.add_argument("--mask_select_method", type=str, default="grammar",
                                choices=['keyword_disconnected', "keyword_connected", "grammar"])
            parser.add_argument("--mask_order_by", type=str, default="dep", choices=['dep', 'pos'])
            parser.add_argument("--keyword_mask", type=str, default="adjacent",
                                choices=['adjacent', 'child', 'child_dep', "na"])
            parser.add_argument("--custom_keywords", type=str, default=['watermarking', 'watermark'])
            parser.add_argument("--message", type=str, default="111")
            parser.add_argument("--spacy_model", type=str, default="en_core_web_sm")
            parser.add_argument("--exclude_cc", type=str2bool, default=True)
        case 'xiaoniu23':
            parser.add_argument("--watermark_type", type=str, default="delta")
            parser.add_argument("--private_key", type=str, default="private key")
            parser.add_argument("--n", type=int, default=5)
        case 'aiwei23b':
            parser.add_argument("--embedding_input_path", type=str, default="~/sok-llm-watermark/watermark_reliability_release/watermarks/aiwei23b/data/sts/train.jsonl")
            parser.add_argument("--embedding_output_path", type=str, default="~/sok-llm-watermark/watermark_reliability_release/watermarks/aiwei23b/data/embeddings/train_embeddings.txt")
            parser.add_argument("--aiwei23b_model_path", type=str, default="~/sok-llm-watermark/watermark_reliability_release/watermarks/aiwei23b/model/compositional-bert-large-uncased")
            parser.add_argument("--aiwei23b_size", type=int, default=2000)
            parser.add_argument("--aiwei23b_output_model", type=str, default="~/sok-llm-watermark/watermark_reliability_release/watermarks/aiwei23b/model/transform_model_cbert.pth")
            parser.add_argument("--watermark_model_epochs", type=int, default=2000)
            parser.add_argument("--watermark_model_lr", type=float, default=0.006)
            parser.add_argument("--aiwei23b_input_dim", type=int, default=1024)
            parser.add_argument("--mapping_length", type=int, default=50257)
            parser.add_argument("--mapping_output_dir", type=str, default="~/sok-llm-watermark/watermark_reliability_release/watermarks/aiwei23b/data/mappings/")
            parser.add_argument("--aiwei23b_watermark_type", type=str, default="context")
            parser.add_argument("--aiwei23b_window_size", type=int, default=0)
            parser.add_argument("--aiwei23b_chunk_size", type=int, default=10)
            parser.add_argument("--aiwei23b_delta", type=int, default=1)
            parser.add_argument("--transform_model", type=str, default="~/sok-llm-watermark/watermark_reliability_release/watermarks/aiwei23b/model/transform_model_cbert.pth")
            parser.add_argument("--embedding_model", type=str, default="~/sok-llm-watermark/watermark_reliability_release/watermarks/aiwei23b/model/compositional-bert-large-uncased")
            parser.add_argument("--aiwei23b_trained", type=str2bool, default=True)
            
            

            ######################################################################
            # Add your code here
            ######################################################################
            # If you have specific arguments for your watermark, add them here
            ######################################################################
            
            

    
    args = parser.parse_args()
    
    
    

    ###########################################################################
    # Argument validation and conditional setting
    ###########################################################################
    # for removing some columns to save space
    args.columns_to_remove = args.columns_to_remove.split(",") if args.columns_to_remove else []

    # if decoding scheme is not sampling, then set generation seed to None
    # to avoid confusion and calling the torch rng unnecessarily
    args.generation_seed = args.generation_seed if args.use_sampling else None

    # -1 value for min_generations means no specified minimum
    # with the assumption that the
    if args.min_generations <= 0:
        args.min_generations = MAX_GENERATIONS
        print(
            f"Warning: min_generations is -1. A hardcoded value of {MAX_GENERATIONS} will be used to limit the generation loop."
        )

    if args.limit_indices is None:
        print("No limit_indices specified, pulling all examples from the dataset.")
    else:
        print(f"Limiting iteration to {args.limit_indices} examples from the dataset.")

    # split wandb tags
    if args.wandb_tags != "":
        args.wandb_tags = args.wandb_tags.split(",")
    else:
        args.wandb_tags = []

    if args.output_dir is None:
        args.output_dir = 'runs/'+str(args.watermark)+'/'+str(args.dataset_name)
    else:
        args.output_dir = 'runs/'+args.output_dir
    main(args)
