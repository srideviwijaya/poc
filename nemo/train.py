import nemo_run as run

from nemo.collections import llm


def configure_recipe(nodes: int = 1, gpus_per_node: int = 1):
    recipe = llm.nemotron3_4b.pretrain_recipe(
        dir="/checkpoints/nemotron", # Path to store checkpoints
        name="nemotron_pretraining",
        tensor_parallelism=1,
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
        max_steps=100, # Setting a small value for the quickstart
    )

    recipe.trainer.val_check_interval = 100
    return recipe

def local_executor_torchrun(nodes: int = 1, devices: int = 1) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "0",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor

def run_pretraining():
    recipe = configure_recipe()
    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)

    executor.ntasks_per_node = 1
    executor.env_vars["CUDA_VISIBLE_DEVICES"] = "0"

    recipe.model.config.num_layers = 8
    recipe.trainer.strategy.tensor_model_parallel_size = 1
    recipe.trainer.devices = 1

    run.run(recipe, executor=executor)

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_pretraining()