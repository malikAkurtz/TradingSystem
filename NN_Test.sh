#!/bin/bash
make clean
make
./out > output.txt
python3 plot_losses.py