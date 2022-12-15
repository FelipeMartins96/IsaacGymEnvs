case $1 in
0) python train_task.py seed=42 task=VSS experiment=ddpg-cma-base;;
1) python train_task.py seed=42 task=VSS train.params.config.gamma=0.95 experiment=ddpg-cma-gamma0.95;;
2) python train_task.py seed=42 task=VSS task.env.has_initial_ball_vel=False experiment=ddpg-cma-noInitBallVel-noEnergy;;
3) python train_task.py seed=42 task=VSS task.env.has_initial_ball_vel=False task.env.has_env_ou_noise=False experiment=ddpg-cma-noInitBallVelAndOU;;
4) python train_task.py seed=43 task=VSS experiment=ddpg-cma-base;;
5) python train_task.py seed=43 task=VSS train.params.config.gamma=0.95 experiment=ddpg-cma-gamma0.95;;
6) python train_task.py seed=43 task=VSS task.env.has_initial_ball_vel=False experiment=ddpg-cma-noInitBallVel-noEnergy;;
7) python train_task.py seed=43 task=VSS task.env.has_initial_ball_vel=False task.env.has_env_ou_noise=False experiment=ddpg-cma-noInitBallVelAndOU;;
8) python train_task.py seed=44 task=VSS experiment=ddpg-cma-base;;
9) python train_task.py seed=44 task=VSS train.params.config.gamma=0.95 experiment=ddpg-cma-gamma0.95;;
10) python train_task.py seed=44 task=VSS task.env.has_initial_ball_vel=False experiment=ddpg-cma-noInitBallVel-noEnergy;;
11) python train_task.py seed=44 task=VSS task.env.has_initial_ball_vel=False task.env.has_env_ou_noise=False experiment=ddpg-cma-noInitBallVelAndOU;;
*) echo "Opcao Invalida!" ;;
esac