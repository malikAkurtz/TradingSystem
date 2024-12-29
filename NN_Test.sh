#!/bin/bash
make clean
make
./bin/trading_system > output.txt
python3 graph_node.py