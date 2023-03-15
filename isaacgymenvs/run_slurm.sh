case $1 in
0) python train.py seed=10 wandb_group=exp018 experiment=ppo-dma task=VSSDecentralizedMA task.env.n_controlled_robots=3;;
1) python train.py seed=20 wandb_group=exp018 experiment=ppo-dma task=VSSDecentralizedMA task.env.n_controlled_robots=3;;
2) python train.py seed=30 wandb_group=exp018 experiment=ppo-dma task=VSSDecentralizedMA task.env.n_controlled_robots=3;;
*) echo "Opcao Invalida!" ;;
esac