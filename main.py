import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX = 960
WINDOWSIZEY = 600

# Initialize pygame
pygame.init()
FONT = pygame.font.Font("freesansbold.ttf", 40)

DISPSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Board (Improved)")

BOUNDINC = 5
WHITE = (255, 255, 255)  
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False

# Load the pre-trained model
MODEL = load_model("FinalTry.keras")


LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 
    34: '8', 35: '9'
}


iswriting = False
PREDICT = True
num_xcord = []
num_ycord = []

image_cnt = 1

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        # Handle mouse drawing
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPSURF, WHITE, (xcord, ycord), 4, 0)
            num_xcord.append(xcord)
            num_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False

            # Ensure there is something drawn
            if len(num_xcord) > 0 and len(num_ycord) > 0:
                # Get bounding box of the drawn digit
                rect_min_x = max(min(num_xcord) - BOUNDINC, 0)
                rect_max_x = min(max(num_xcord) + BOUNDINC, WINDOWSIZEX)
                rect_min_y = max(min(num_ycord) - BOUNDINC, 0)
                rect_max_y = min(max(num_ycord) + BOUNDINC, WINDOWSIZEY)

                num_xcord = []
                num_ycord = []

                # Get the drawn area as an array
                img_arr = pygame.surfarray.array3d(DISPSURF)[rect_min_x:rect_max_x, rect_min_y:rect_max_y]
                img_arr = np.mean(img_arr, axis=2)  # Convert to grayscale
                img_arr = img_arr.T  # Transpose to match the correct orientation

                if IMAGESAVE:
                    cv2.imwrite(f"image_{image_cnt}.png", img_arr)
                    image_cnt += 1

                if PREDICT:
                    # Resize and normalize the image to fit the model input
                    image = cv2.resize(img_arr, (28, 28))
                    image = np.pad(image, ((10, 10), (10, 10)), 'constant', constant_values=0)
                    image = cv2.resize(image, (28, 28)) / 255.0

                    # Predict the digit
                    prediction = MODEL.predict(image.reshape(1, 28, 28, 1))
                    label = str(LABELS[np.argmax(prediction)])

                    # Render the label
                    textSurface = FONT.render(label, True, RED, WHITE)
                    textRecObj = textSurface.get_rect()
                    textRecObj.left = rect_min_x
                    textRecObj.bottom = rect_min_y - 10

                    # Draw a rectangle around the digit
                    pygame.draw.rect(DISPSURF, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2)

                    # Blit the label to the screen
                    DISPSURF.blit(textSurface, textRecObj)

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPSURF.fill(BLACK)  # Clear the screen with black color

    pygame.display.update()
