case $1 in
0) python train.py task=VSS seed=42 experiment=ppo-base-50k;;
1) python train.py task=VSS seed=43 experiment=ppo-base-50k;;
2) python train.py task=VSS seed=44 experiment=ppo-base-50k;;
3) python train.py task=VSS seed=42 task.env.n_controlled_robots=3 experiment=ppo-cma-base-50k;;
4) python train.py task=VSS seed=43 task.env.n_controlled_robots=3 experiment=ppo-cma-base-50k;;
5) python train.py task=VSS seed=44 task.env.n_controlled_robots=3 experiment=ppo-cma-base-50k;;
*) echo "Opcao Invalida!" ;;
esac