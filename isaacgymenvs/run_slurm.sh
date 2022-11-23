case $1 in
0) python train.py task=VSS experiment=ppo-sim2realWeights;;
*) echo "Opcao Invalida!" ;;
esac