import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

UPLOAD_DIR = "uploaded_files"

# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Check if the file is an image (optional)
    if not file.content_type.startswith('image'):
        return JSONResponse(status_code=415, content={"message": "Only image files are allowed."})

    # Save the uploaded file to disk (optional)
    with open(f"{UPLOAD_DIR}/{file.filename}", "wb") as buffer:
        buffer.write(await file.read())

    return {"filename": file.filename}
