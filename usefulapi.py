from fastapi import FastAPI, Query, HTTPException
import openai
import os
import subprocess
import json
import sqlite3
import requests
import shutil
import duckdb
import markdown
from pathlib import Path
from git import Repo
from bs4 import BeautifulSoup
from PIL import Image
import librosa
import soundfile as sf
import pandas as pd
import shlex

app = FastAPI()

# Set your OpenAI API key here or use environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

def run_command(command: list):
    try:
        sanitized_command = shlex.split(" ".join(command))
        result = subprocess.run(sanitized_command, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"Error executing command: {e.stderr}")

def validate_path(path: str):
    if not path.startswith("/data/"):
        raise HTTPException(status_code=400, detail="Access to paths outside /data/ is not allowed.")

@app.post("/run")
async def run_task(task: str = Query(..., description="Task description")):
    try:
        if "fetch data from an API" in task:
            url = task.split("fetch data from an API and save it ")[1]
            response = requests.get(url)
            with open("/data/api_data.json", "wb") as f:
                f.write(response.content)
        elif "clone a git repo" in task:
            repo_url = task.split("clone a git repo and make a commit ")[1]
            repo_path = "/data/repo"
            if not os.path.exists(repo_path):
                Repo.clone_from(repo_url, repo_path)
            repo = Repo(repo_path)
            repo.git.commit("--allow-empty", "-m", "Automated commit")
        elif "run a SQL query" in task:
            db_type = "sqlite" if "SQLite" in task else "duckdb"
            db_path = "/data/database.db" if db_type == "sqlite" else "/data/database.duckdb"
            validate_path(db_path)
            query = task.split("run a SQL query on ")[1]
            conn = sqlite3.connect(db_path) if db_type == "sqlite" else duckdb.connect(db_path)
            result = conn.execute(query).fetchall()
            conn.close()
            with open("/data/query_result.json", "w") as f:
                json.dump(result, f)
        elif "extract data from" in task:
            url = task.split("extract data from (i.e. scrape) ")[1]
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            with open("/data/scraped_data.txt", "w") as f:
                f.write(soup.get_text())
        elif "compress or resize an image" in task:
            image_path = "/data/image.png"
            validate_path(image_path)
            img = Image.open(image_path)
            img = img.resize((800, 600))
            img.save("/data/image_resized.png", optimize=True)
        elif "transcribe audio" in task:
            audio_path = "/data/audio.mp3"
            validate_path(audio_path)
            y, sr = librosa.load(audio_path, sr=None)
            sf.write("/data/audio.wav", y, sr)
            response = openai.Audio.transcribe("whisper-1", audio=open("/data/audio.wav", "rb"))
            with open("/data/transcription.txt", "w") as f:
                f.write(response["text"])
        elif "convert Markdown to HTML" in task:
            md_path = "/data/document.md"
            validate_path(md_path)
            with open(md_path) as f:
                html_content = markdown.markdown(f.read())
            with open("/data/document.html", "w") as f:
                f.write(html_content)
        elif "write an API endpoint that filters a CSV file" in task:
            csv_path = "/data/data.csv"
            validate_path(csv_path)
            df = pd.read_csv(csv_path)
            filtered_df = df[df["column"] > 10]
            filtered_df.to_json("/data/filtered_data.json", orient="records")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/read")
async def read_file(path: str = Query(..., description="File path to read")):
    validate_path(path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found.")
    
    try:
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
        return {"status": "success", "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
