case $1 in
0) python train.py seed=10 wandb_group=exp016 experiment=ppo-cma task=VSS task.env.n_controlled_robots=3;;
1) python train.py seed=20 wandb_group=exp016 experiment=ppo-cma task=VSS task.env.n_controlled_robots=3;;
2) python train.py seed=30 wandb_group=exp016 experiment=ppo-cma task=VSS task.env.n_controlled_robots=3;;
*) echo "Opcao Invalida!" ;;
esac