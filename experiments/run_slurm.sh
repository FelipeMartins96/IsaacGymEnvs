case $1 in
0) python train_task.py --record --num-envs=4096 --experiment=ddpg-base;;
*) echo "Opcao Invalida!" ;;
esac