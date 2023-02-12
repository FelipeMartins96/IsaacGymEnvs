case $1 in
0) python train.py task=VSS seed=40 experiment=ppo wandb_group=teste;;
1) python train.py task=VSS seed=40 task.env.n_controlled_robots=3 experiment=ppo-cma wandb_group=teste;;
2) python train.py task=VSSDecentralizedMA seed=40 experiment=ppo-dma wandb_group=teste;;
*) echo "Opcao Invalida!" ;;
esac