case $1 in
0) python train_task.py --num-envs=4096 --experiment=ddpg-sim2realWeights;;
*) echo "Opcao Invalida!" ;;
esac