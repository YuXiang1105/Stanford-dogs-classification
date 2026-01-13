@echo off

if not exist venv\Scripts\activate (
    echo creating virtual environment
    python -m venv venv
    call venv\Scripts\activate
    echo virtual environment created
) else (
    echo virtual environmnet already exists
)





