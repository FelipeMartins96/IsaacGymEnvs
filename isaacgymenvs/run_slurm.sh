case $1 in
0) python train.py task=VSS experiment=ppo-base;;
*) echo "Opcao Invalida!" ;;
esac