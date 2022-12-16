case $1 in
0) python train.py task=VSS seed=42 experiment=ppo-base;;
1) python train.py task=VSS seed=43 experiment=ppo-base;;
2) python train.py task=VSS seed=44 experiment=ppo-base;;
3) python train.py task=VSS seed=42 task.env.n_controlled_robots=3 experiment=ppo-cma-base;;
4) python train.py task=VSS seed=43 task.env.n_controlled_robots=3 experiment=ppo-cma-base;;
5) python train.py task=VSS seed=44 task.env.n_controlled_robots=3 experiment=ppo-cma-base;;
*) echo "Opcao Invalida!" ;;
esac