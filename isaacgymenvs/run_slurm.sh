case $1 in
0) python train.py task=VSS seed=40 experiment=ppo wandb_group=ppo-noMove;;
1) python train.py task=VSS seed=41 experiment=ppo wandb_group=ppo-noMove;;
2) python train.py task=VSS seed=42 experiment=ppo wandb_group=ppo-noMove;;
3) python train.py task=VSS seed=43 experiment=ppo wandb_group=ppo-noMove;;
4) python train.py task=VSS seed=44 experiment=ppo wandb_group=ppo-noMove;;
5) python train.py task=VSS seed=40 task.env.n_controlled_robots=3 experiment=ppo-cma wandb_group=ppo-cma-noMove;;
6) python train.py task=VSS seed=41 task.env.n_controlled_robots=3 experiment=ppo-cma wandb_group=ppo-cma-noMove;;
7) python train.py task=VSS seed=42 task.env.n_controlled_robots=3 experiment=ppo-cma wandb_group=ppo-cma-noMove;;
8) python train.py task=VSS seed=43 task.env.n_controlled_robots=3 experiment=ppo-cma wandb_group=ppo-cma-noMove;;
9) python train.py task=VSS seed=44 task.env.n_controlled_robots=3 experiment=ppo-cma wandb_group=ppo-cma-noMove;;
10) python train.py task=VSSDecentralizedMA seed=40 experiment=ppo-dma wandb_group=ppo-dma-noMove;;
11) python train.py task=VSSDecentralizedMA seed=41 experiment=ppo-dma wandb_group=ppo-dma-noMove;;
12) python train.py task=VSSDecentralizedMA seed=42 experiment=ppo-dma wandb_group=ppo-dma-noMove;;
13) python train.py task=VSSDecentralizedMA seed=43 experiment=ppo-dma wandb_group=ppo-dma-noMove;;
14) python train.py task=VSSDecentralizedMA seed=44 experiment=ppo-dma wandb_group=ppo-dma-noMove;;
*) echo "Opcao Invalida!" ;;
esac