case $1 in
0) python train.py task=VSS experiment=3v3VSS42;;
1) python train.py task=VSS experiment=3v3VSS43 seed=43;;
2) python train.py task=VSS experiment=3v3VSS44 seed=44;;
*) echo "Opcao Invalida!" ;;
esac