import os
import tempfile
import uuid
import zipfile
import shutil
from lxml import etree
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional, Generator, List, Dict, Any, Callable, Set
import copy # Needed for deep copying elements
import re # For parsing adjustment values
from pathlib import Path # For easier path manipulation
import requests
import json
import firebase_admin
from datetime import timedelta
import traceback
import zipfile
import math
from lxml import etree
from pptx import Presentation
from pptx.shapes.group import GroupShape
from pptx.shapes.graphfrm import GraphicFrame
from pptx.enum.shapes import MSO_SHAPE_TYPE
from firebase_admin import credentials, storage, db

db_url = 'https://snb-ai-translation-agent-default-rtdb.firebaseio.com'
secret = 'nAWmdbcHRL9UGDOP0S1Rp0pZ2c7TEIapUrsEBzHJ'
download_folder = "downloads"
# Path to your service account key
cred = credentials.Certificate("service_account_key.json")

firebase_initialized = False

def init_firebase():
    global firebase_initialized
    if not firebase_initialized:
        try:
        # Initialize app with storage bucket
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'snb-ai-translation-agent.firebasestorage.app',
                'databaseURL': db_url
            })
            firebase_initialized = True
        except Exception as e:
            pass


class Logger:
    def __init__(self, id: str = "0"):
        self.publish_ref = f"logs/{id}"

    def info(self, message):
        print(f"INFO: {message}")

    def debug(self, message):
        print(f"DEBUG: {message}")

    def warning(self, message):
        self.print_and_write(f"WARNING: {message}")

    def error(self, message):
        self.print_and_write(f"ERROR: {message}")

    def exception(self, message):
        self.print_and_write(f"EXCEPTION: {message}")

    def print_and_write(self, message):
        print(message)
        if not os.path.exists("logs"):
            os.makedirs("logs")
        with open(f"{self.publish_ref}.txt", "a") as f:
            f.write(message + "\n")

    def publish(self, message):
        init_firebase()
        try:
            ref = db.reference(f"/{self.publish_ref}")
            # Push a new object (auto-generates a unique key)
            ref.push({
                'message': str(message)
            })
        except Exception as e:
            self.exception(e)
        print(message)


def upload_output(output_path: str):
    init_firebase()
    bucket = storage.bucket()
    blob = bucket.blob(output_path)

    # Upload from local file
    blob.upload_from_filename(output_path)

    # Optionally make it publicly accessible
    return blob.generate_signed_url(expiration=timedelta(hours=2))
    blob.make_public()
    return blob.public_url


def download_file(url: str, filename: str):
    response = requests.get(url)
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    output_path = f"{download_folder}/{filename}"
    # Save the content to a file
    with open(output_path, 'wb') as file:
        file.write(response.content)
    return output_path


def clear_id(id: str):
    requests.put(
        f"{db_url}/translation_requests/{id}.json?auth={secret}",
        data=json.dumps({})
    )
