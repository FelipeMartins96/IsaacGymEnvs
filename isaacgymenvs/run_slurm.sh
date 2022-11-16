case $1 in
0) python train.py task=VSS experiment=VSS;;
1) python train.py task=VSS experiment=VSS+noEnvOU has_env_ou_noise=False;;
2) python train.py task=VSS experiment=VSS+noInitialBallVel task.env.has_initial_ball_vel=False;;
3) python train.py task=VSS experiment=VSS+noInitialBallVel+noEnvOU task.env.has_initial_ball_vel=False has_env_ou_noise=False;;
4) python train.py task=VSS experiment=VSS+noInitialBallVel+noEnvOU+noGrad task.env.has_initial_ball_vel=False has_env_ou_noise=False task.env.has_grad=False;;
*) echo "Opcao Invalida!" ;;
esac