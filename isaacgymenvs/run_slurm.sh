case $1 in
0) python train.py task=VSS seed=10 experiment=ppo wandb_group=exp009;;
1) python train.py task=VSS seed=10 experiment=ppo-cma wandb_group=exp009 task.env.n_controlled_robots=3;;
2) python train.py task=VSSDecentralizedMA seed=10 experiment=ppo-dma wandb_group=exp009;;
3) python train.py task=VSS seed=20 experiment=ppo wandb_group=exp009;;
4) python train.py task=VSS seed=20 experiment=ppo-cma wandb_group=exp009 task.env.n_controlled_robots=3;;
5) python train.py task=VSSDecentralizedMA seed=20 experiment=ppo-dma wandb_group=exp009;;
6) python train.py task=VSS seed=30 experiment=ppo wandb_group=exp009;;
7) python train.py task=VSS seed=30 experiment=ppo-cma wandb_group=exp009 task.env.n_controlled_robots=3;;
8) python train.py task=VSSDecentralizedMA seed=30 experiment=ppo-dma wandb_group=exp009;;
*) echo "Opcao Invalida!" ;;
esac