## Clone repo
```
git clone https://github.com/67Samuel/alexnet-tiny-imagenet-1.git alexnet-tiny
cd alexnet-tiny
```

## Download data
```
chmod +x prep.sh
./prep.sh
```

## To train a single model
```
python train.py
```

## To do a wandb hyperparameter sweep
```
git checkout -b wandb_sweep origin/wandb_sweep
wandb sweep bayes_sweep.yaml
```
Then, copy and paste the command similar to wandb agent 67samuel/alexnet-tiny/YOUR-SWEEP-ID

Link to report:
https://app.wandb.ai/67samuel/from_scratch/reports/Analysis-of-SNIP--VmlldzoxODIxMTA

