import cv2
from tkinter import *
from PIL import ImageTk, Image
from srcs.model.model import *


def make_label(parent, img, side="left", fill="both"):
    label = Label(parent, image=img, bg="white")
    label.pack(side=side, fill=fill)


def make_text(parent, txt, **kwargs):
    text = Text(parent, kwargs)
    text.pack()
    text.insert(END, txt)


root = Tk()
root.geometry("800x400+240+140")
root.title("Classifier")

capture = cv2.VideoCapture(0)
while True:
    output, image = capture.read()
    cv2.imshow('SMILE FACE', image)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:  # escape
        break
    if k == 13:  # enter
        filename = "capture.jpg"
        cv2.imwrite(filename, image)
        break
capture.release()
cv2.destroyAllWindows()

image = ImageTk.PhotoImage(Image.open(filename).resize((400, 300)))
make_label(root, image)

"""
Вот тут должна быть часть с моделькой
"""

output = "LETTER"  # output = model(filename)
make_text(root, output, bd=0, height=10, width=50, padx=120, pady=180, font=("Verdana", 18, "bold"), relief=RIDGE)

root.protocol("WM_DELETE_WINDOW", root.destroy)
root.mainloop()