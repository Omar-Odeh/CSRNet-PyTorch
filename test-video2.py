import torch
import torchvision.transforms as transforms
from model import CSRNet
import cv2


# Load the model
model = CSRNet()
model = model.cuda()
checkpoint = torch.load('partBmodel_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()


# Define transformer to preprocess the input image
transform = transforms.Compose([
    transforms.ToTensor(),            # Convert to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # Normalize pixel values
        std=[0.229, 0.224, 0.225]
    )
])


# Capture the video
cap = cv2.VideoCapture(
    './ext_testing/videos/mixkit-crowd-of-skiers-in-the-top-of-the-mountain-31941-medium.mp4')

if (cap.isOpened() == False):
    print("Error opening video")

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        # Preprocess the input frame
        input_frame = transform(frame).unsqueeze(0).cuda()

        # Run the input through the model
        with torch.no_grad():
            output = model(input_frame)

        # Retrieve the output
        count = output.cpu().sum().item()
        countText = f"Count : {int(count)}"

        print(countText, end="\r")

        # cv2.putText(frame, countText, (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 50), 2, cv2.LINE_AA)

        # Display the frame
        # cv2.imshow("frame", frame)
        # cv2.waitKey()

        # Quit when q is pressed
        # if cv2.waitKey(1) == ord('q'):
        #     break

    else:
        break

cap.release()
# cv2.destroyAllWindows()

