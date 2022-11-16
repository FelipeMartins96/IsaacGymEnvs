python train.py task=VSS task.env.has_grad=False experiment=VSS headless=True max_iterations=1000;
python train.py task=VSS task.env.has_grad=True experiment=VSS+grad headless=True max_iterations=1000
case $1 in
0) python train.py task=VSS task.env.has_grad=False experiment=VSS max_iterations=1000; ;;
*) echo "Opcao Invalida!" ;;
esac