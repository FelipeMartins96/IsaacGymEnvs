case $1 in
0) python train.py task=VSS experiment=ppo-base;;
1) python train.py task=VSS train.params.config.gamma=0.95 experiment=ppo-gamma0.95;;
2) python train.py task=VSS task.env.has_initial_ball_vel=False experiment=ppo-noInitBallVel;;
3) python train.py task=VSS task.env.has_initial_ball_vel=False task.env.has_env_ou_noise=False experiment=ppo-noInitBallVelAndOU;;
*) echo "Opcao Invalida!" ;;
esac