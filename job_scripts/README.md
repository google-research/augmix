# USE SRUN FOR DEBUGGING
```
srun -p gpu --gres=gpu:1 -t 23:59:59 --ntasks=1 --cpus-per-task=32 --mem=50G --pty bash
poetry shell
python augmix_refactored/script/cifar.py --model wrn
```