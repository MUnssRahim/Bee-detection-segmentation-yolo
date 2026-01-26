import requests
import cv2
import numpy as np
import os

# CONFIGURATION
API_URL = "http://127.0.0.1:8000/detect"
TEST_IMAGE = r"C:\Users\HP\Desktop\Bees\Mid_Defence_DetectionModel\WhatsApp Image 2026-01-21 at 11.21.35 AM (1).jpeg"  # <--- Make sure this image exists!

def run_visual_test():
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Error: Could not find image '{TEST_IMAGE}'")
        return

    print(f"📡 Sending '{TEST_IMAGE}' to API...")

    try:
        # 1. SEND REQUEST
        with open(TEST_IMAGE, "rb") as f:
            response = requests.post(API_URL, files={"file": f})

        if response.status_code == 200:
            print("✅ Success! displaying result...")
            
            # 2. CONVERT BYTES TO IMAGE
            # The API sends back raw bytes. We need to turn them back into an image.
            image_bytes = np.frombuffer(response.content, np.uint8)
            decoded_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

            # 3. SHOW WINDOW
            # This creates a window named "Bee Detection Result"
            cv2.imshow("Bee Detection Result", decoded_image)

            # 4. WAIT FOR EXIT
            print("Press any key (or close the window) to exit.")
            cv2.waitKey(0)        # Waits forever until a key is pressed
            cv2.destroyAllWindows() # Closes the window
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("\n❌ CONNECTION REFUSED!")
        print("💡 Solution: You must run 'uvicorn api:app' in a separate terminal window first.")

if __name__ == "__main__":
    run_visual_test()