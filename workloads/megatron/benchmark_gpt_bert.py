import argparse
import json
from datetime import datetime

from util import run_cmd

from collections import namedtuple

UniformParallelArgs = namedtuple("UniformParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "dp", "op", "pp",
    "force_batch_dim_mapping"
])

BenchmarkCase = namedtuple("BenchmarkCase", [
    "batch_size", "model_config", "num_micro_batches", "parallel_mode",
    "parallel_args"
])

GPTModelConfig = namedtuple(
    "GPTModelConfig",
    ["seq_len", "hidden_size", "num_layers", "num_heads", "vocab_size"])

gpt_specs = {
    #                      S，   H,   L,  head,   V,
    "125M": GPTModelConfig(1024, 768, 12, 12, 51200),
    "350M": GPTModelConfig(1024, 1024, 24, 16, 51200),
    "760M": GPTModelConfig(1024, 1536, 24, 16, 51200),
    "1.3B": GPTModelConfig(1024, 2048, 24, 32, 51200),
    "2.6B": GPTModelConfig(1024, 2560, 32, 32, 51200),
    "6.7B": GPTModelConfig(1024, 4096, 32, 32, 51200),
    "15B": GPTModelConfig(1024, 5120, 48, 40, 51200),
    "39B": GPTModelConfig(1024, 8192, 48, 64, 51200),
    "76B": GPTModelConfig(1024, 10240, 60, 80, 51200),
}

# Temporary debug suite
# key = the number of gpus, value = a list of cases
# B, model, NB, PM, (RS, Remat, 3D Config, FM)
tmp_suite = {
    1: [
        BenchmarkCase(16, gpt_specs["350M"], 1, "uniform",
                      UniformParallelArgs(True, True, 1, 1, 1, True))
    ],
    8: [
        BenchmarkCase(128, GPTModelConfig(1024, 4096, 4, 32, 51200),
                      4, "uniform",
                      UniformParallelArgs(True, True, 4, 1, 2, True)),
    ],
    2 :[
        BenchmarkCase(32, gpt_specs["350M"], 2, "uniform",
                      UniformParallelArgs(True, True, 2, 1, 1, True))
    ],
}

benchmark_suites = {
    "gpt.tmp": tmp_suite,
    #"gpt.grid_search_manual": suite_manual_gpt.grid_search_manual,
}

def benchmark_all(args):
    num_gpus = args.nproc_per_node * args.nnodes

    try:
        _ = benchmark_suites[args.suite][num_gpus]
    except KeyError:
        print(f"No available benchmark suite for {args.suite} with {num_gpus} GPUs.")
        exit()
    output_name = args.exp_name + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model = args.suite.split(".")[0]

    for case in benchmark_suites[args.suite][num_gpus]:
        case = tuple(tuple(x) if isinstance(x, tuple) else x for x in case)
        case_str = str((model,) + case)

        if args.nnodes == 1:
            # Single node
            ret = run_cmd('python -m torch.distributed.launch '
                         f'--nproc_per_node {args.nproc_per_node} '
                         'benchmark_gpt_bert_one_case.py '
                          f'"{case_str}" '
                          f'{output_name}')
        else:
            # Multiple nodes
            ret = run_cmd('python -m torch.distributed.launch '
                         f'--nproc_per_node {args.nproc_per_node} '
                         f'--nnodes {args.nnodes} '
                         f'--node_rank {args.node_rank} '
                         f'--master_addr {args.master_addr} '
                         f'--master_port {args.master_port} '
                         'benchmark_gpt_bert_one_case.py '
                         f'"{case_str}" '
                         f'{output_name}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int)
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=str)
    parser.add_argument("--suite", type=str, default="gpt.tmp")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--g_batch_size", type=int, default=-1)
    parser.add_argument("--repeat", type=int, default=-1)
    parser.add_argument('--scheduler_ip', type=str)
    parser.add_argument('--scheduler_port', type=int, default=6889)
    parser.add_argument('--trainer_port', type=int)
    parser.add_argument('--job_id', type=int, default=-1)
    args = parser.parse_args()

    configs = {
        'gbs': args.g_batch_size,
        'repeat': args.repeat,
        'scheduler_ip': args.scheduler_ip,
        'scheduler_port': args.scheduler_port,
        'trainer_port': args.trainer_port,
        'job_id': args.job_id,
    }
    with open('./configs.json', 'w') as f:
        json.dump(configs, f)

    benchmark_all(args)
