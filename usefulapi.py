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

openai.api_key = os.environ("eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDQzMTlAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.4_QfcYiQAB7DTHMtUMyNprvu3Xz6ui-upXP17FFgEUM")

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
        if "install uv and run" in task:
            run_command(["pip", "install", "uv"])  # Ensure uv is installed
            run_command(["python", "datagen.py", "23f2004319@ds.study.iitm.ac.in.email"])
        elif "format /data/format.md" in task:
            run_command(["npx", "prettier@3.4.2", "--write", "/data/format.md"])
        elif "count the number of Wednesdays" in task:
            with open("/data/dates.txt", "r") as f:
                dates = [line.strip() for line in f]
            count = sum(1 for date in dates if datetime.datetime.strptime(date, "%Y-%m-%d").weekday() == 2)
            with open("/data/dates-wednesdays.txt", "w") as f:
                f.write(str(count))
        elif "sort contacts in /data/contacts.json" in task:
            with open("/data/contacts.json", "r") as f:
                contacts = json.load(f)
            contacts.sort(key=lambda x: (x["last_name"], x["first_name"]))
            with open("/data/contacts-sorted.json", "w") as f:
                json.dump(contacts, f, indent=4)
        elif "write first line of recent .log files" in task:
            log_files = sorted(Path("/data/logs").glob("*.log"), key=os.path.getmtime, reverse=True)[:10]
            with open("/data/logs-recent.txt", "w") as f:
                for log in log_files:
                    with open(log, "r") as lf:
                        f.write(lf.readline())
        elif "extract H1 titles from Markdown files" in task:
            index = {}
            for md_file in Path("/data/docs").glob("*.md"):
                with open(md_file, "r") as f:
                    for line in f:
                        if line.startswith("#"):
                            index[md_file.name] = line.strip("# ")
                            break
            with open("/data/docs/index.json", "w") as f:
                json.dump(index, f, indent=4)
        elif "extract sender’s email address" in task:
            with open("/data/email.txt", "r") as f:
                email_text = f.read()
            response = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "user", "content": f"Extract the sender's email address: {email_text}"}])
            sender_email = response["choices"][0]["message"]["content"].strip()
            with open("/data/email-sender.txt", "w") as f:
                f.write(sender_email)
        elif "extract credit card number" in task:
            image = Image.open("/data/credit-card.png")
            response = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "Extract the credit card number from this image."}], files=[("image", image)])
            card_number = response["choices"][0]["message"]["content"].strip()
            with open("/data/credit-card.txt", "w") as f:
                f.write(card_number)
        elif "find the most similar pair of comments" in task:
            with open("/data/comments.txt", "r") as f:
                comments = [line.strip() for line in f]
            response = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "Find the two most similar comments in this list."}], temperature=0.0)
            similar_comments = response["choices"][0]["message"]["content"].strip()
            with open("/data/comments-similar.txt", "w") as f:
                f.write(similar_comments)
        elif "calculate total sales of Gold tickets" in task:
            conn = sqlite3.connect("/data/ticket-sales.db")
            cursor = conn.cursor()
            cursor.execute("SELECT SUM(price * units) FROM tickets WHERE type = 'Gold'")
            total_sales = cursor.fetchone()[0]
            conn.close()
            with open("/data/ticket-sales-gold.txt", "w") as f:
                f.write(str(total_sales))
        elif "fetch data from an API" in task:
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
