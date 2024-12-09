# Create a video using multiple static plots output as .png files
#
# (c) 2024 Liquid Instruments Pty. Ltd.
# Last edited on 2 December 2024

import cv2 
import os

image_folder = './images' #relative path to images to stitch together as a video
video_name = './Radar-{}.mp4'.format('Decreasing SNR')
num_frames = 20


# Store individual frames into list
images = []
cnt=num_frames
while cnt>0:
    images.append("image"+str(cnt)+".png")
    cnt-=1

# Create frame dimension and store its shape dimensions
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# cv2's VideoWriter object will create a frame 
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc,1, (width,height))

# Create the video from individual images using for loop
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# Close all the frames
cv2.destroyAllWindows()

# Release the video write object
video.release()