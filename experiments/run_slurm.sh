case $1 in
0) python train_task.py task=VSS seed=42 experiment=ddpg-base-50k;;
1) python train_task.py task=VSS seed=43 experiment=ddpg-base-50k;;
2) python train_task.py task=VSS seed=44 experiment=ddpg-base-50k;;
3) python train_task.py task=VSS seed=42 task.env.n_controlled_robots=3 experiment=ddpg-cma-base-50k;;
4) python train_task.py task=VSS seed=43 task.env.n_controlled_robots=3 experiment=ddpg-cma-base-50k;;
5) python train_task.py task=VSS seed=44 task.env.n_controlled_robots=3 experiment=ddpg-cma-base-50k;;
*) echo "Opcao Invalida!" ;;
esac