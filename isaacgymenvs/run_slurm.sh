case $1 in
0) python train.py task=VSS experiment=ppo-$1 seed=$1;;
1) python train.py task=VSS experiment=ppo-$1 seed=$1;;
2) python train.py task=VSS experiment=ppo-$1 seed=$1;;
*) echo "Opcao Invalida!" ;;
esac