case $1 in
0) python train_task.py seed=42 task=VSS experiment=ddpg-base;;
1) python train_task.py seed=42 task=VSS train.params.config.gamma=0.95 experiment=ddpg-gamma0.95;;
2) python train_task.py seed=42 task=VSS task.env.has_initial_ball_vel=False experiment=ddpg-noInitBallVel;;
3) python train_task.py seed=42 task=VSS task.env.has_initial_ball_vel=False task.env.has_env_ou_noise=False experiment=ddpg-noInitBallVelAndOU;;
4) python train_task.py seed=43 task=VSS experiment=ddpg-base;;
5) python train_task.py seed=43 task=VSS train.params.config.gamma=0.95 experiment=ddpg-gamma0.95;;
6) python train_task.py seed=43 task=VSS task.env.has_initial_ball_vel=False experiment=ddpg-noInitBallVel;;
7) python train_task.py seed=43 task=VSS task.env.has_initial_ball_vel=False task.env.has_env_ou_noise=False experiment=ddpg-noInitBallVelAndOU;;
8) python train_task.py seed=44 task=VSS experiment=ddpg-base;;
9) python train_task.py seed=44 task=VSS train.params.config.gamma=0.95 experiment=ddpg-gamma0.95;;
10) python train_task.py seed=44 task=VSS task.env.has_initial_ball_vel=False experiment=ddpg-noInitBallVel;;
11) python train_task.py seed=44 task=VSS task.env.has_initial_ball_vel=False task.env.has_env_ou_noise=False experiment=ddpg-noInitBallVelAndOU;;
*) echo "Opcao Invalida!" ;;
esac