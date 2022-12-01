case $1 in
0) python train_task.py --num-envs=4096 --experiment=ddpg-increaseMove-$1 --wandb --seed=$1;;
1) python train_task.py --num-envs=4096 --experiment=ddpg-increaseMove-$1 --wandb --seed=$1;;
2) python train_task.py --num-envs=4096 --experiment=ddpg-increaseMove-$1 --wandb --seed=$1;;
*) echo "Opcao Invalida!" ;;
esac