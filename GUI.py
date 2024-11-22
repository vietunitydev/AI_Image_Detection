import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import random
import Image_Detector  # Ensure this module is available

root = tk.Tk()
root.title("AI Detector")
root.geometry("1000x588+100+100")
root.configure(bg="#DDDDDD")
root.resizable(False, False)


def showimage():
    filename = filedialog.askopenfilename(initialdir=os.getcwd(),
                                          title='Select Image File',
                                          filetypes=(
                                          ('PNG file', '*.png'), ('JPG file', '*.jpg'), ('JPEG file', '*.jpeg')))
    if filename:
        img = Image.open(filename)
        img = img.resize((400, 438), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        lbl.configure(image=img, width=400, height=438)
        lbl.image = img
        root.selected_img_path = filename  # Corrected variable name


# Display output
def display_output():
    image_path = root.selected_img_path  # Corrected variable
    model_path = 'Model_Trained/19-11/trained_model.keras'

    try:
        output1, output2 = Image_Detector.classify_image(image_path, model_path)
        if output1 == "AI":
            tk.Label(frame, text=output1, font="arial 30 bold", bg="#FF9999", fg="#FF0000", padx=70, pady=9).place(x=160, y=182)
            tk.Label(frame, text=round(output2, 4), font="arial 30 bold", bg="#FF9999").place(x=184, y=360)
        else:
            tk.Label(frame, text=output1, font="arial 30 bold", bg="#FF9999", fg="#00FF00", padx=30, pady=9).place(x=150, y=182)
            tk.Label(frame, text=round(output2, 4), font="arial 30 bold", bg="#FF9999").place(x=184, y=360)
    except Exception as e:
        print("Error in classification:", e)


# Icon
try:
    image_icon = tk.PhotoImage(file="logo.png")
    root.iconphoto(False, image_icon)
except Exception as e:
    print("Icon loading error:", e)

tk.Label(root, width=150, height=13, bg="#FF0000").pack()

# Frame
frame = tk.Frame(root, width=875, height=463, bg="#FFF")
frame.place(x=62, y=62)

# Logo
try:
    logo = tk.PhotoImage(file="logo.png")
    tk.Label(frame, image=logo, bg="#FFF").place(x=19, y=9)
except Exception as e:
    print("Logo loading error:", e)

background_label = tk.Label(frame, bg="#FFF")
background_label.place(x=113, y=15)

tk.Label(background_label, text="AI DETECTOR", font="Tahoma 31 bold italic", bg="white", padx=10, pady=10).pack()
tk.Label(frame, text="Classify:", font="arial 31 bold", bg="#FFF").place(x=31, y=119)
tk.Label(frame, text="Reliability:", font="arial 31 bold", bg="#FFF").place(x=31, y=281)

# Result boxes
square1 = tk.Frame(frame, width=300, height=75, bg="#FF9999", bd=2, relief="solid", highlightbackground="black")
square1.place(x=100, y=180)

square2 = tk.Frame(frame, width=300, height=75, bg="#FF9999", bd=2, relief="solid", highlightbackground="black")
square2.place(x=100, y=350)

# Image selection
selectimage = tk.Frame(frame, width=425, height=438, bg="#FF9999")
selectimage.place(x=438, y=10)

f = tk.Frame(selectimage, bd=4, bg="black", width=400, height=350, relief=tk.GROOVE)
f.place(x=12, y=12)

lbl = tk.Label(f, bg="black")
lbl.place(x=0, y=0)

# Buttons
tk.Button(selectimage, text="Select image", width=13, height=1, font="arial 17 bold", command=showimage).place(x=11,
                                                                                                               y=375)
tk.Button(selectimage, text="Detect", width=13, height=1, font="arial 17 bold", command=display_output).place(x=222,
                                                                                                              y=375)

root.mainloop()