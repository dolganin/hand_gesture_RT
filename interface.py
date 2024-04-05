import cv2

img_id = 0
shot = 1
result = []
capture = cv2.VideoCapture(0)
while True:
    image = capture.read()[1]
    cv2.imshow('SMILE FACE', image)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:  # escape
        break
    if not shot % 60:
        filename = "capture_%i.jpg" % img_id
        cv2.imwrite(filename, image)
        # output = model(filename)
        if output == "del" and img_id > 0:
            result.pop(img_id-1)
        result.append(output.replace("space", " ").replace("del", ""))
        img_id += 1
    shot += 1
capture.release()
cv2.destroyAllWindows()

with open("output.txt", "w") as file:
    file.write("".join(list(dict.fromkeys(result))))