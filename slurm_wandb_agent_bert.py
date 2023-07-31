#!/usr/bin/env python
## example of usage: python slurm_wandb_agent_bert.py 82m6orwn -e jongsong -p "BERT with Speical Tokens"
import argparse
import os
from pathlib import Path


def main(args):
    slurm_dir = Path(args.slurm_dir)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    slurm_gitignore = Path(slurm_dir / '.gitignore')

    if not slurm_gitignore.exists():
        with open(slurm_gitignore, 'w') as f:
            f.write('*')

    template_path = dict()
    template_path['srun.sh'] = Path(slurm_dir / args.srun_template)
    template_path['sbatch.sh'] = Path(slurm_dir / args.sbatch_template)
    template_path['sbatch_run.sh'] = Path(slurm_dir / args.sbatch_run_template)


    template_str = dict()
    template_str['srun.sh'] = '\n'.join((
        "#!/bin/bash",
    ))
    for fname in ['srun.sh', 'sbatch.sh', 'sbatch_run.sh']:
        if not template_path[fname].exists():
            with open(template_path[fname], 'w') as f:
                f.write(template_str[fname])
        os.system(f"chmod +x {template_path[fname]}")

    job_dir = Path(slurm_dir / args.sweep_id)
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if not args.force:
            raise ValueError(
                f'Got sweep_id = {args.sweep_id}, but directory {job_dir} already exists!'
            )

    file_path = dict()
    file_path['srun.sh'] = Path(job_dir / args.srun_filename)
    file_path['sbatch.sh'] = Path(job_dir / args.sbatch_filename)
    file_path['sbatch_run.sh'] = Path(job_dir / args.sbatch_run_filename)

    # The following might be nice but are definitely not necessary:
    # TODO: Allow SBATCH defaults to be overridden by pass-through arguments to this script

    file_str = dict()
    complete_sweep_id_entires = []
    if args.wandb_entity:
        complete_sweep_id_entires.append(args.wandb_entity)
    if args.wandb_project:
        complete_sweep_id_entires.append(args.wandb_project)
    complete_sweep_id_entires.append(args.sweep_id)
    complete_sweep_id = "/".join(complete_sweep_id_entires)
    file_str['srun.sh'] = "\n".join((
        "",
        f"output=`wandb agent --count 20 {complete_sweep_id}`",
        "run_id=${output: (-11):8}",
        "val_acc=${output: (-2):2}",
        f'echo "$output"',
        f'echo "$run_id"',
        f'echo "$val_acc"',
        f"if  ((10#$val_acc!=0))" , #adjust val_acc to rerun
        f"then",
        f"  sbatch {file_path['sbatch_run.sh']} $run_id",
        f"else",
        f"  sbatch {file_path['sbatch.sh']}",
        f"fi"
        # f"pwd", # for testing
        # f"./test.sh", # for testing
        ## 
    ))
    # sbatch throws the job which runs iteratively
    file_str['sbatch.sh'] = "\n".join((
        "",
        f"#SBATCH --job-name={args.sweep_id}",
        f"#SBATCH --output={job_dir}/%A-%a.out",
        # f"#SBATCH --error={job_dir}/%A-%a.error",
        f"#SBATCH --gres=gpu:4",
        f"#SBATCH --nodes=1",
        f"#SBATCH --time=0-12:00:00",
        f"#SBATCH --mem=64000MB",
        f"#SBATCH --cpus-per-task=8",
        f"#SBATCH --partition=P1",
    #  f'#SBATCH --array=1-{args.num_jobs}\n' if args.num_jobs > 1 else '',
        f"bash {file_path['srun.sh']}"
    ))
        # f"srun {file_path['srun_initial.sh']}"))
        # f"iterations=2" # 총 몇 번이나 연속으로 돌릴 것인지
        # f"jobid=$(sbatch --parsable srun.sh)"
        # f"for((i=0; i<$iterations; i++)); do"           
        #     f'dependency="afterany:${jobid}"'
        #     f'echo "dependency: $dependency"'
        #     f"jobid=$(sbatch --parsable --dependency=$dependency srun.sh)"
        #     f'dependency=",${dependency}afterany:${jobid}"'
        # f'done'
        ## val_acc를 어케받지?
    file_str['sbatch_run.sh'] = "\n".join((
        "#!/bin/bash",
        f"#SBATCH --job-name={args.sweep_id}",
        f"#SBATCH --output={job_dir}/%A-%a.out",
        # f"#SBATCH --error={job_dir}/%A-%a.error",
        f"#SBATCH --gres=gpu:4",
        f"#SBATCH --nodes=1",
        f"#SBATCH --time=0-12:00:00",
        f"#SBATCH --mem=64000MB",
        f"#SBATCH --cpus-per-task=8",
        f"#SBATCH --partition=P1",
        # add sweep parametersㅐㅕㅅ
        f"lr=`grep -o \'\"learning_rate\": [^,]*\' ./models/zeshel/crossencoder/$1/training_params.json | grep -o \'[^ ]*$\'`",
        f"dim_red=`grep -o \'\"dim_red\": [^,]*\' ./models/zeshel/crossencoder/$1/training_params.json | grep -o \'[^ ]*$\'`",
        f"layers=`grep -o \'\"layers\": [^,]*\' ./models/zeshel/crossencoder/$1/training_params.json | grep -o \'[^ ]*$\'`",
        f"train_batch_size=`grep -o \'\"train_batch_size\": [^,]*\' ./models/zeshel/crossencoder/$1/training_params.json | grep -o \'[^ ]*$\'`",
        f"decoder=`grep -o \'\"decoder\": [^,]*\' ./models/zeshel/crossencoder/$1/training_params.json | grep -o \'[^ ]*$\'`",
        "echo \"lr=${lr}, dim_red=${dim_red}, layers=${layers}, train_batch_size=${train_batch_size}, decoder=${decoder}\"",
        "python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --decoder=${decoder} --dim_red=${dim_red} --layers=${layers} --learning_rate=${lr} --train_batch_size=${train_batch_size} --resume=True --run_id=$1",
        # choose to re-run
        "val_acc=${output: (-2):2}",
        f"if  ((10#$val_acc!=0))" , #adjust val_acc to rerun
        f"then",
        f"  sbatch {file_path['sbatch_run.sh']} $run_id",
        f"else",
        f"  sbatch {file_path['sbatch.sh']}",
        f"fi"
        #    f"json=`cat ./models/zeshel/crossencoder/$1/training_params.json`",
        # f"echo $json"
        # f"echo $json | grep -o \'\"learning_rate\":\"[^\"]*' | grep -o \'[^\"]*$\'"
        # f"bash {file_path['srun.sh']}"
    ))
    for fname in ['srun.sh', 'sbatch.sh', 'sbatch_run.sh']:
        os.system(f"cp {template_path[fname]} {file_path[fname]}")
        with open(file_path[fname], 'a') as f:
            f.write(file_str[fname])
    # elif args.special_token_bert:
    #     file_str['srun.sh'] = "\n".join((
    #         "",
    #         f"output=`python blink/crossencoder/train_cross.py --learning_rate 2e-05 --num_train_epochs 1000 --train_batch_size 128 --eval_batch_size 128 --wandb 'BERT with Speical Tokens' --save True --train_size 10000 --valid_size 10000 --architecture special_token --add_linear True`",
    #         f'echo "$output"',
    #         f'echo "$run_id"',
    #         f"sbatch {file_path['sbatch.sh']}",
    #         ## 
    #     ))
    #     file_str['sbatch.sh'] = "\n".join((
    #         "",
    #         f"#SBATCH --job-name={args.sweep_id}",
    #         f"#SBATCH --output={job_dir}/%A-%a.out",
    #         f"#SBATCH --gres=gpu:4",
    #         f"#SBATCH --nodes=1",
    #         f"#SBATCH --time=0-12:00:00",
    #         f"#SBATCH --mem=96000MB",
    #         f"#SBATCH --cpus-per-task=8",
    #         f"#SBATCH --partition=P1",
    #         f"bash {file_path['srun.sh']}"
    #     ))

    #     for fname in ['srun.sh', 'sbatch.sh']:
    #         os.system(f"cp {template_path[fname]} {file_path[fname]}")
    #         with open(file_path[fname], 'a') as f:
    #             f.write(file_str[fname])

    if args.edit_srun:
        os.system(f"{args.editor} {file_path['srun.sh']}")

    if args.edit_sbatch:
        os.system(f"{args.editor} {file_path['sbatch.sh']}")

    if args.edit_sbatch_run:
        os.system(f"{args.editor} {file_path['sbatch_run.sh']}")


    if not args.no_run:
        os.system(f"sbatch {file_path['sbatch.sh']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run multiple wandb agents via SLURM')
    parser.add_argument('sweep_id', type=str)
    parser.add_argument('-e', '--wandb-entity', type=str, help="your wandb username or team name")
    parser.add_argument('-p', '--wandb-project', type=str, help="your wandb project for this sweep.")
    parser.add_argument('--num-jobs', type=int, default=1)
    parser.add_argument('--num-agents-per-job', type=int, default=1)
    parser.add_argument(
        '--edit-sbatch',
        action='store_true',
        help="open the sbatch.sh file in a text editor before running")
    parser.add_argument(
        '--edit-sbatch-run',
        action='store_true',
        help="open the sbatch_run.sh file in a text editor before running")
    parser.add_argument(
        '--edit-srun',
        action='store_true',
        help="open the srun.sh file in a text editor before running")
    # Which architecture to run


    parser.add_argument(
        '--no-run',
        action='store_true',
        help="don't run the job(s), just create the sh files")
    parser.add_argument(
        '-f',
        '--force',
        action='store_true',
        help="force creation of job folder if it already exists, potentially overwriting data"
    )
    parser.add_argument(
        '--slurm_dir',
        type=str,
        default="slurm_output",
        help="directory to store slurm files in")
    parser.add_argument(
        '--srun-filename',
        type=str,
        default="srun.sh",
        help="filename of srun script (to be created)")
    parser.add_argument(
        '--sbatch-filename',
        type=str,
        default="sbatch.sh",
        help="filename of sbatch script (to be created)")
    parser.add_argument(
        '--sbatch-run-filename',
        type=str,
        default="sbatch_run.sh",
        help="filename of sbatch script (to be created)")
    parser.add_argument(
        '--srun-template',
        type=str,
        default="srun.sh",
        help="filename of srun template (which will be copied and appended to) relative to SLURM_DIR"
    )
    parser.add_argument(
        '--sbatch-template',
        type=str,
        default="sbatch.sh",
        help="filename of sbatch template (which will be copied and appended to) relative to SLURM_DIR"
    )
    parser.add_argument(
        '--sbatch-run-template',
        type=str,
        default="sbatch_run.sh",
        help="filename of sbatch template (which will be copied and appended to) relative to SLURM_DIR"
    )
    parser.add_argument(
        '-q',
        '--slurm-queue',
        type=str,
        default="titanx-short",
        help="slurm partition/queue"
    )
    parser.add_argument('--editor', type=str, default="vim")
    # More generally, should be able to use EDITOR = "${EDITOR:-vim}" but I don't always have my editor set.

    args = parser.parse_args()
    main(args)