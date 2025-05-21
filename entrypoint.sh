#!/bin/bash
if [ "$1" = "ui" ]; then
  exec streamlit run app/streamlit_app.py
else
  exec python cli/query_cli.py "$@"
fi