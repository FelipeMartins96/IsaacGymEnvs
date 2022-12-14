case $1 in
0) python train_task.py task=VSS experiment=ddpg-base;;
1) python train_task.py task=VSS train.params.config.gamma=0.95 experiment=ddpg-gamma0.95;;
2) python train_task.py task=VSS task.env.has_initial_ball_vel=False experiment=ddpg-noInitBallVel;;
3) python train_task.py task=VSS task.env.has_initial_ball_vel=False task.env.has_env_ou_noise=False experiment=ddpg-noInitBallVelAndOU;;
*) echo "Opcao Invalida!" ;;
esac