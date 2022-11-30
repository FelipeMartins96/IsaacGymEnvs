case $1 in
0) python train.py task=VSS experiment=ppo-testRender capture_video=True headless=False;;
*) echo "Opcao Invalida!" ;;
esac