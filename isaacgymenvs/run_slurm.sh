case $1 in
0) python train.py task=VSS seed=40 experiment=ppo-40;;
1) python train.py task=VSS seed=41 experiment=ppo-41;;
2) python train.py task=VSS seed=42 experiment=ppo-42;;
3) python train.py task=VSS seed=43 experiment=ppo-43;;
4) python train.py task=VSS seed=44 experiment=ppo-44;;
5) python train.py task=VSS seed=40 task.env.n_controlled_robots=3 experiment=ppo-dma-40;;
6) python train.py task=VSS seed=41 task.env.n_controlled_robots=3 experiment=ppo-dma-41;;
7) python train.py task=VSS seed=42 task.env.n_controlled_robots=3 experiment=ppo-dma-42;;
8) python train.py task=VSS seed=43 task.env.n_controlled_robots=3 experiment=ppo-dma-43;;
9) python train.py task=VSS seed=44 task.env.n_controlled_robots=3 experiment=ppo-dma-44;;
10) python train.py task=VSSDecentralizedMA seed=40 experiment=ppo-cma-40;;
11) python train.py task=VSSDecentralizedMA seed=41 experiment=ppo-cma-41;;
12) python train.py task=VSSDecentralizedMA seed=42 experiment=ppo-cma-42;;
13) python train.py task=VSSDecentralizedMA seed=43 experiment=ppo-cma-43;;
14) python train.py task=VSSDecentralizedMA seed=44 experiment=ppo-cma-44;;
*) echo "Opcao Invalida!" ;;
esac