case $1 in
3) python train.py task=VSS seed=42 task.env.n_controlled_robots=3 experiment=ppo-cma-adaptative-100k;;
4) python train.py task=VSS seed=43 task.env.n_controlled_robots=3 experiment=ppo-cma-adaptative-100k;;
5) python train.py task=VSS seed=44 task.env.n_controlled_robots=3 experiment=ppo-cma-adaptative-100k;;
*) echo "Opcao Invalida!" ;;
esac