#!/usr/bin/env bash
/home/human_studies/.pyenv/versions/3.12.9/bin/uwsgi --http :9999 --module pyolimp_human_studies:app --master --processes 4 --threads 4
