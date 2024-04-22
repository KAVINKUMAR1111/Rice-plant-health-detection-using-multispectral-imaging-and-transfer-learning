import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import uvicorn
import tensorflow as tf
app = FastAPI()
model = tf.keras.models.load_model('unet-model')
def predict_image(img_array):
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    generated_image = predictions[0]
    if generated_image.shape[-1] == 1:
        generated_image = np.squeeze(generated_image, axis=-1)
    return generated_image
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = image.load_img(io.BytesIO(contents), target_size=(512, 512))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    generated_image = predict_image(img_array)
    img_bytes = io.BytesIO()
    plt.imsave(img_bytes, generated_image, format='png')
    img_bytes.seek(0)
    return StreamingResponse(img_bytes, media_type="image/png")
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5001)