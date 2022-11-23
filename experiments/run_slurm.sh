case $1 in
0) python train_task.py --num-envs=4096 --experiment=ddpg-WmoveWenergy;;
*) echo "Opcao Invalida!" ;;
esac