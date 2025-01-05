#!/bin/bash
make clean
make
./bin/trading_system
python3 graph_node.py
