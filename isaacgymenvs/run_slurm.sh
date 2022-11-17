case $1 in
0) python train.py task=VSS experiment=3v3-noOU-noBallVel task.env.has_initial_ball_vel=False task.env.has_env_ou_noise=False;;
*) echo "Opcao Invalida!" ;;
esac