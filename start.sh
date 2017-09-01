#!/usr/bin/env bash

cd /opt/deepdetect/build/main && (nohup ./dede -host 0.0.0.0 &) && python /home/objdetect.py
