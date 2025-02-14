#!/usr/bin/env bash
gunicorn --workers 1 --bind 0.0.0.0:9999 pyolimp_human_studies:app
