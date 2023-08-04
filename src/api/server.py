import uvicorn
import io
from fastapi import FastAPI, UploadFile
from fastapi.responses import RedirectResponse
from .prediction import read_imagefile, predict

app = FastAPI(title="Image Classifier API")

@app.get("/")
def redirect_swagger():
    return RedirectResponse("/docs")

@app.post("/predict")
async def predict_api(file: UploadFile) -> None:      
    upload = io.BytesIO(await file.read())
    image = read_imagefile(upload)
    prediction = predict(image)
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, debug=True)
