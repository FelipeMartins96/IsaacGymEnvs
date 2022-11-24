case $1 in
0) python train.py task=VSS experiment=ppo-fixMoveW;;
*) echo "Opcao Invalida!" ;;
esac