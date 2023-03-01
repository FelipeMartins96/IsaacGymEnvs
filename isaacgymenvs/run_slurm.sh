case $1 in
0) python train.py seed=10 wandb_group=exp014 experiment=ppo        task=VSS                train.params.config.entropy_coef=0.001 task.env.has_move=False;;
1) python train.py seed=10 wandb_group=exp014 experiment=ppo-cma    task=VSS                train.params.config.entropy_coef=0.001 task.env.has_move=False  task.env.n_controlled_robots=3;;
2) python train.py seed=10 wandb_group=exp014 experiment=ppo-dma    task=VSSDecentralizedMA train.params.config.entropy_coef=0.001 task.env.has_move=False;;
3) python train.py seed=20 wandb_group=exp014 experiment=ppo        task=VSS                train.params.config.entropy_coef=0.001 task.env.has_move=False;;
4) python train.py seed=20 wandb_group=exp014 experiment=ppo-cma    task=VSS                train.params.config.entropy_coef=0.001 task.env.has_move=False  task.env.n_controlled_robots=3;;
5) python train.py seed=20 wandb_group=exp014 experiment=ppo-dma    task=VSSDecentralizedMA train.params.config.entropy_coef=0.001 task.env.has_move=False;;
6) python train.py seed=30 wandb_group=exp014 experiment=ppo        task=VSS                train.params.config.entropy_coef=0.001 task.env.has_move=False;;
7) python train.py seed=30 wandb_group=exp014 experiment=ppo-cma    task=VSS                train.params.config.entropy_coef=0.001 task.env.has_move=False  task.env.n_controlled_robots=3;;
8) python train.py seed=30 wandb_group=exp014 experiment=ppo-dma    task=VSSDecentralizedMA train.params.config.entropy_coef=0.001 task.env.has_move=False;;
*) echo "Opcao Invalida!" ;;
esac