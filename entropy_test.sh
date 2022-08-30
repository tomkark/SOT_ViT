for i in `seq 0.025 0.025 1.0`; do
  python main.py --ot $i --model cct_2 --conv-size 3 --conv-layers 2 --print-freq -1 --epochs 30 --workers 1 cifar10/
done