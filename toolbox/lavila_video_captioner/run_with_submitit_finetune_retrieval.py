# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
#
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.
# """
# A script to run multinode training with submitit.
# """
# import argparse
# import os
# import uuid
# from pathlib import Path
#
# import toolbox.lavila_video_captioner.main_finetune_retrieval as main_finetune
# import submitit
#
#
# def parse_args():
#     parser = main_finetune.get_args_parser()
#     parser = argparse.ArgumentParser("Submitit for lavila fine-tuning", parents=[parser])
#     parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
#     parser.add_argument("--nodes", default=8, type=int, help="Number of nodes to request")
#     parser.add_argument("--timeout", default=2880, type=int, help="Duration of the job")
#     parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
#
#     parser.add_argument("--partition", default="learnlab", type=str, help="Partition where to submit")
#     parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
#     parser.add_argument('--comment', default="", type=str,
#                         help='Comment to pass to scheduler, e.g. priority message')
#     return parser.parse_args()
#
#
# def get_shared_folder() -> Path:
#     user = os.getenv("USER")
#     if Path("/checkpoint/").is_dir():
#         p = Path(f"/checkpoint/{user}/experiments/lavila_ft")
#         p.mkdir(exist_ok=True)
#         return p
#     raise RuntimeError("No shared folder available")
#
#
# def get_init_file():
#     # Init file must not exist, but it's parent dir must exist.
#     os.makedirs(str(get_shared_folder()), exist_ok=True)
#     init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
#     if init_file.exists():
#         os.remove(str(init_file))
#     return init_file
#
#
# class Trainer(object):
#     def __init__(self, args):
#         self.args = args
#
#     def __call__(self):
#         import main_finetune_retrieval as main_finetune
#
#         self._setup_gpu_args()
#         main_finetune.main(self.args)
#
#     def checkpoint(self):
#         import submitit
#
#         self.args.dist_url = get_init_file().as_uri()
#         print("Requeuing ", self.args)
#         empty_trainer = type(self)(self.args)
#         return submitit.helpers.DelayedSubmission(empty_trainer)
#
#     def _setup_gpu_args(self):
#         import submitit
#         from pathlib import Path
#
#         job_env = submitit.JobEnvironment()
#         self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
#         self.args.gpu = job_env.local_rank
#         self.args.rank = job_env.global_rank
#         self.args.world_size = job_env.num_tasks
#         print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
#
#
# def main():
#     args = parse_args()
#     if args.job_dir == "":
#         args.job_dir = get_shared_folder() / "%j"
#
#     # Note that the folder will depend on the job_id, to easily track experiments
#     executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)
#
#     num_gpus_per_node = args.ngpus
#     nodes = args.nodes
#     timeout_min = args.timeout
#
#     partition = args.partition
#     kwargs = {}
#     if args.use_volta32:
#         kwargs['slurm_constraint'] = 'volta32gb'
#     if args.comment:
#         kwargs['slurm_comment'] = args.comment
#
#     executor.update_parameters(
#         mem_gb=40 * num_gpus_per_node,
#         gpus_per_node=num_gpus_per_node,
#         tasks_per_node=num_gpus_per_node,  # one task per GPU
#         cpus_per_task=10,
#         nodes=nodes,
#         timeout_min=timeout_min,  # max is 60 * 72
#         # Below are cluster dependent parameters
#         slurm_partition=partition,
#         slurm_signal_delay_s=120,
#         **kwargs
#     )
#
#     executor.update_parameters(name="lavila_ft")
#
#     args.dist_url = get_init_file().as_uri()
#     args.output_dir = args.job_dir
#
#     trainer = Trainer(args)
#     job = executor.submit(trainer)
#
#     print("Submitted job_id:", job.job_id)
#
#
# if __name__ == "__main__":
#     main()
