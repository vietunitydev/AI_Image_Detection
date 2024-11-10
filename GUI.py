import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import random
import image_detector

root = Tk()
root.title("AI Detector")
root.geometry("1000x588+100+100") 
root.configure(bg="#DDDDDD")
root.resizable(False, False)

def showimage():
    filename = filedialog.askopenfilename(initialdir=os.getcwd(),
                                            title='Select Image File', filetypes=(('PNG file', '*.png'),
                                                                                    ('JPG file', '*.jpg'),  
                                                                                    ('JPEG file', '*.jpeg')))
    if filename:  
        img = Image.open(filename)
        img = img.resize((400, 438), Image.LANCZOS)  
        img = ImageTk.PhotoImage(img)
        lbl.configure(image=img, width=400, height=438)  
        lbl.image = img

        root.selected_img_path = filename

#get picture
def get_output():
    return random.choice(["AI", "Human"])

#get reliability
def get_reliability():
    return random.uniform(0, 1)

#display output
def display_output():

    image_path = root.selected_image
    model_path = '/Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Image_Classifier/trained_model.keras'

    output1, output2 = image_detector.classify_image(image_path, model_path)

    if output1 == "AI":
        Label(frame, text=output1, font="arial 30 bold", bg="#FF9999", fg="#FF0000").place(x=225, y=190)
        Label(frame, text=round(output2, 4), font="arial 30 bold", bg="#FF9999").place(x=184, y=360)
    else:
        Label(frame, text=output1, font="arial 30 bold", bg="#FF9999", fg="#00FF00").place(x=182, y=190)
        Label(frame, text=round(output2, 4), font="arial 30 bold", bg="#FF9999").place(x=184, y=360)

# Icon
image_icon = PhotoImage(file="logo.png")
root.iconphoto(False, image_icon)

Label(root, width=150, height=13, bg="#FF0000").pack()  

# Frame
frame = Frame(root, width=875, height=463, bg="#FFF")  
frame.place(x=62, y=62)

logo = PhotoImage(file="D:/Linh ta linh tinh/GUI/logo.png")
Label(frame, image=logo, bg="#FFF").place(x=19, y=9)  

background_label = Label(frame, bg="#FFF")
background_label.place(x=113, y=15)

Label(background_label, text="AI DETECTOR", font="Tahoma 31 bold italic", bg="white", padx=10, pady=10).pack()
Label(frame, text="Classify:", font="arial 31 bold", bg="#FFF").place(x=31, y=119)  
Label(frame, text="Reliability:", font="arial 31 bold", bg="#FFF").place(x=31, y=281) 

# Result box
square1 = Frame(frame, width=300, height=75, bg="#FF9999", bd=2, relief="solid", highlightbackground="black")
square1.place(x=100, y=180)

square2 = Frame(frame, width=300, height=75, bg="#FF9999", bd=2, relief="solid", highlightbackground="black")
square2.place(x=100, y=350)

# Select image
selectimage = Frame(frame, width=425, height=438, bg="#FF9999")  
selectimage.place(x=438, y=10)

f = Frame(selectimage, bd=4, bg="black", width=400, height=350, relief=GROOVE)  
f.place(x=12, y=12)

lbl = Label(f, bg="black")
lbl.place(x=0, y=0)

#Button
Button(selectimage, text="Select image", width=13, height=1, font="arial 17 bold", command=showimage).place(x=11, y=375)  
Button(selectimage, text="Detect", width=13, height=1, font="arial 17 bold", command=display_output).place(x=222, y=375)  

root.mainloop()