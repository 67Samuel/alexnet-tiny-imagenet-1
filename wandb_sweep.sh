import subprocess

wandb sweep wandb_sweep.yaml

batcmd="dir"
sweep_id = subprocess.check_output(batcmd, shell=True)
wandb agent sweep_id
