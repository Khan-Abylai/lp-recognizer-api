#!/bin/bash
/usr/bin/python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 9001 --workers 4