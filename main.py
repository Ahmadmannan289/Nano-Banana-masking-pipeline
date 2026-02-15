import os
import shutil
import uuid
from io import BytesIO
import fal_client
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from PIL import Image


app = FastAPI()

# Create an 'uploads' directory if it doesn't exist
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

def combine_images_with_mask(original_path, mask_path, output_path):
    """
    Combines an original image with a mask image and saves the result.

    This function assumes the original and mask images have the same dimensions.
    The mask is treated as an alpha channel, where colored pixels (like the red
    drawing) will be semi-transparently overlaid on the original image.
    
    Args:
        original_path (str): The file path to the original image.
        mask_path (str): The file path to the mask image.
        output_path (str): The file path to save the combined image.
    """
    try:
        # Open the original and mask images
        original_image = Image.open(original_path).convert("RGBA")
        mask_image = Image.open(mask_path).convert("RGBA")

        # Ensure both images have the same size
        if original_image.size != mask_image.size:
            print("Error: The original image and mask image must have the same dimensions.")
            return

        # Create a new image for the combined result
        combined_image = original_image.copy()
        
        # Paste the mask on top of the original image
        # The mask's alpha channel will determine the transparency of the drawing
        combined_image.alpha_composite(mask_image)

        # Save the final combined image
        combined_image.save(output_path)
        print(f"Successfully combined images and saved to {output_path}")

    except FileNotFoundError:
        print("Error: One of the files was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def save_file(image: UploadFile):
    file_extension = os.path.splitext(image.filename)[1]
    processed_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOADS_DIR, processed_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    return file_path

def recontext_masked_area(combined_image_path, prompt):
    """
    Uploads the combined image to FAL, sends it to nano-banana edit model,
    and overwrites the image with the edited result.
    """

    print("Uploading image to FAL:", combined_image_path)

    image_url = fal_client.upload_file(combined_image_path)

    print("FAL image URL:", image_url)
    print("Prompt:", prompt)

    # Submit job
    handler = fal_client.submit(
        "fal-ai/nano-banana-pro/edit",
        arguments={
            "prompt": f'"{prompt}. Do not add the outline in the final image."',
            "image_urls": [image_url],
            "resolution": "4K",
            "aspect_ratio": "auto",
        }
    )
    
#     handler = fal_client.submit(
#     "fal-ai/qwen-image-edit-plus",
#     arguments={
#         "prompt": f'"{prompt}. Do not add the outline in the final image."',
#         "image_urls": [image_url],
#         # "image_size": "landscape_16_9", ### square_hd, square, portrait_4_3, portrait_16_9, landscape_4_3, landscape_16_9
#         "num_inference_steps": 30,
#         "guidance_scale": 2.5,
        
#     },
#     webhook_url="https://optional.webhook.url/for/results",
# )


    request_id = handler.request_id
    print("Request ID:", request_id)


    result = fal_client.result(
        "fal-ai/nano-banana-pro/edit",
        request_id
    )

    print("FAL result:", result)

    # Download and overwrite the image
    if "images" in result and len(result["images"]) > 0:
        output_image_url = result["images"][0]["url"]

        import requests
        response = requests.get(output_image_url)
        response.raise_for_status()

        with open(combined_image_path, "wb") as f:
            f.write(response.content)

        print("Image successfully updated:", combined_image_path)
    else:
        raise RuntimeError("No images returned from FAL API")

    return combined_image_path



@app.post("/process")
async def process_image(image: UploadFile = File(...), mask: UploadFile = File(...), prompt: str = Form(None)):
    
    mask_path = save_file(mask)
    image_path = save_file(image)    

    combined_image_path = os.path.join(UPLOADS_DIR, f"{uuid.uuid4()}.png")

    combine_images_with_mask(image_path, mask_path, combined_image_path)

    recontext_masked_area(combined_image_path, prompt)

    return JSONResponse(content={"processed_image_url": f"/{combined_image_path}"})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():

    return Response(status_code=204)

@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.get("/{file_path:path}")
async def get_static(file_path: str):
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
