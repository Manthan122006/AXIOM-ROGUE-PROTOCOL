#!/bin/bash
cd "$(dirname "$0")"
pip3 install -q pandas numpy scikit-learn
( sleep 12; open "http://127.0.0.1:8000" ) &
python3 server.py
