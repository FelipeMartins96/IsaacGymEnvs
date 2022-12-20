case $1 in
0) python train.py task=VSS seed=40 experiment=ppo;;
1) python train.py task=VSS seed=41 experiment=ppo;;
2) python train.py task=VSS seed=42 experiment=ppo;;
3) python train.py task=VSS seed=43 experiment=ppo;;
4) python train.py task=VSS seed=44 experiment=ppo;;
5) python train.py task=VSS seed=40 task.env.n_controlled_robots=3 experiment=ppo-dma;;
6) python train.py task=VSS seed=41 task.env.n_controlled_robots=3 experiment=ppo-dma;;
7) python train.py task=VSS seed=42 task.env.n_controlled_robots=3 experiment=ppo-dma;;
8) python train.py task=VSS seed=43 task.env.n_controlled_robots=3 experiment=ppo-dma;;
9) python train.py task=VSS seed=44 task.env.n_controlled_robots=3 experiment=ppo-dma;;
10) python train.py task=VSSDecentralizedMA seed=40 experiment=ppo-cma;;
11) python train.py task=VSSDecentralizedMA seed=41 experiment=ppo-cma;;
12) python train.py task=VSSDecentralizedMA seed=42 experiment=ppo-cma;;
13) python train.py task=VSSDecentralizedMA seed=43 experiment=ppo-cma;;
14) python train.py task=VSSDecentralizedMA seed=44 experiment=ppo-cma;;
*) echo "Opcao Invalida!" ;;
esac