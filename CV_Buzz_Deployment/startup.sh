#!/bin/bash
pip uninstall -y opencv-python
gunicorn -w 4 -k uvicorn.workers.UvicornWorker API:app