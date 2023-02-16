case $1 in
0) python train.py task=VSS seed=10 experiment=ppo wandb_group=exp002;;
1) python train.py task=VSS seed=20 experiment=ppo wandb_group=exp002;;
2) python train.py task=VSS seed=30 experiment=ppo wandb_group=exp002;;
3) python train.py task=VSS seed=10 experiment=ppo-cma wandb_group=exp002 task.env.n_controlled_robots=3;;
4) python train.py task=VSS seed=20 experiment=ppo-cma wandb_group=exp002 task.env.n_controlled_robots=3;;
5) python train.py task=VSS seed=30 experiment=ppo-cma wandb_group=exp002 task.env.n_controlled_robots=3;;
6) python train.py task=VSSDecentralizedMA seed=10 experiment=ppo-dma wandb_group=exp002;;
7) python train.py task=VSSDecentralizedMA seed=20 experiment=ppo-dma wandb_group=exp002;;
8) python train.py task=VSSDecentralizedMA seed=30 experiment=ppo-dma wandb_group=exp002;;
*) echo "Opcao Invalida!" ;;
esac