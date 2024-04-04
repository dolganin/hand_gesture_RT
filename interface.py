import cv2

id = 0
result = ""
capture = cv2.VideoCapture(0)
while True:
    image = capture.read()[1]
    cv2.imshow('SMILE FACE', image)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:  # escape
        break
    if k == 13:  # enter
        filename = "capture_%i.jpg" % id
        cv2.imwrite(filename, image)
        output = ""
        # output = model(filename)
        if output == "del" and id > 0:
            result = result[:id-1]
        result += output.replace("space", " ").replace("del", "")
        id += 1
capture.release()
cv2.destroyAllWindows()

with open("output.txt", "w") as file:
    file.write(result)