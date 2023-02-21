case $1 in
0) python train.py task=VSS seed=10 experiment=ppo wandb_group=exp010;;
1) python train.py task=VSS seed=10 experiment=ppo-cma wandb_group=exp010 task.env.n_controlled_robots=3;;
2) python train.py task=VSSDecentralizedMA seed=10 experiment=ppo-dma wandb_group=exp010;;
3) python train.py task=VSS seed=20 experiment=ppo wandb_group=exp010;;
4) python train.py task=VSS seed=20 experiment=ppo-cma wandb_group=exp010 task.env.n_controlled_robots=3;;
5) python train.py task=VSSDecentralizedMA seed=20 experiment=ppo-dma wandb_group=exp010;;
6) python train.py task=VSS seed=30 experiment=ppo wandb_group=exp010;;
7) python train.py task=VSS seed=30 experiment=ppo-cma wandb_group=exp010 task.env.n_controlled_robots=3;;
8) python train.py task=VSSDecentralizedMA seed=30 experiment=ppo-dma wandb_group=exp010;;
*) echo "Opcao Invalida!" ;;
esac