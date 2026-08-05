# Launch the Streamlit dashboard using the project's virtual environment.
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\env\Scripts\Activate.ps1
streamlit run app.py
