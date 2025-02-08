import pandas as pd

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from joblib import load


def lifespan(app: FastAPI):
    app.state.model = load("heart_model.joblib")

    yield

    print("Приложение остановлено!")

app = FastAPI(
    description="Heart Disease API",
    lifespan=lifespan
)

@app.post("/predict/")
async def model_calculate(file: UploadFile=File(...)):
    try:
        data = pd.read_json(file.file, orient="split")

        pred = app.state.model.predict(data)

        return JSONResponse(content={"disease": pred.tolist()}, status_code=200)
    
    except Exception as e:
        return JSONResponse(content=f"error: {e}", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)