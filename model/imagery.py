import os
import replicate
import requests 

# Make sure you've replaced 'your_api_token_here' with your actual token
os.environ["REPLICATE_API_TOKEN"] = "r8_KtvdkgngLUs7F4UGIeUXljjB2cu6jWF4fNz9I"

def generate_image():
    prompt = input("")

    output = replicate.run(
        "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
        input={
            "width": 768,
            "height": 768,
            "prompt": prompt,
            "scheduler": "K_EULER",
            "num_outputs": 1,
            "guidance_scale": 7.5,
            "num_inference_steps": 50
        }
    )

    # Extract the first image URL from the output
    image_url = output[0]  

    # Download the image
    response = requests.get(image_url, stream=True)
    response.raise_for_status()  # Raise an exception for bad response codes

    # Create the 'image' directory if it doesn't exist
    os.makedirs("image", exist_ok=True)

    # Save the image
    filename = os.path.basename(image_url)  # Get the filename from the URL
    filepath = os.path.join("image", filename)
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(1024):
            f.write(chunk)

    print(f"Image downloaded and saved as {filepath}")

if __name__ == "__main__":
    generate_image()
