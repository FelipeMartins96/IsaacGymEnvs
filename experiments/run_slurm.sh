case $1 in
0) python train_task.py --num-envs=4096 --experiment=ddpg-testRender --record --wandb;;
*) echo "Opcao Invalida!" ;;
esac