case $1 in
0) python train_task.py task=VSS seed=42 experiment=ddpg-base;;
1) python train_task.py task=VSS seed=43 experiment=ddpg-base;;
2) python train_task.py task=VSS seed=44 experiment=ddpg-base;;
3) python train_task.py task=VSS seed=42 task.env.n_controlled_robots=3 experiment=ddpg-cma-base;;
4) python train_task.py task=VSS seed=43 task.env.n_controlled_robots=3 experiment=ddpg-cma-base;;
5) python train_task.py task=VSS seed=44 task.env.n_controlled_robots=3 experiment=ddpg-cma-base;;
*) echo "Opcao Invalida!" ;;
esac