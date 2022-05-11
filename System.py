

import tkinter as tk
from tkinter import Y, messagebox as tkMessageBox
from tkinter import scrolledtext
from random import shuffle

Start = True

width_app = 480
height_app = 320,

def helloCallBack():
    tkMessageBox.showinfo( "Hello Python", "Hello World")

def click_button_intput_pass():
    for widgets in frame.winfo_children():
        widgets.destroy()

    root = tk.Tk()
    root.title("My GUI")
    root.geometry("480x320")

    frame_input_pass = tk.Frame(root)
    frame_input_pass.pack(side="top", expand=True, fill="both")

    lbl = tk.Label(frame_input_pass, text="Tài Khoản", font=("Arial Bold", 15))
    #Xác định vị trí của label
    lbl.place(x=315, y=5)

    lbl = tk.Label(frame_input_pass, text="Mật Khẩu ", font=("Arial Bold", 15))
    #Xác định vị trí của label
    lbl.place(x=315, y=5)
    

    Button_setup = tk.Button(frame_input_pass, text ="Cài Đặt", command = helloCallBack, height=5, width=30)
    Button_setup.place(x = 10,y = 170)


    


def draw_postion_palm(mycanvas):
    ####clear man hinh
    for widgets in frame.winfo_children():
        widgets.destroy()

    
    mycanvas = tk.Canvas(
        mycanvas,
        height=320,
        width=480,
        bg="#fff"
        )     
    mycanvas.pack()
    mycanvas.create_rectangle(
        240, 2, 477, 317)

    mycanvas.create_rectangle(
        2, 2, 477, 317)

    mycanvas.create_rectangle(
        240, 40, 477, 280)
    mycanvas.create_oval(
        240, 40, 477, 280, width= 5)

    Button_intput_pass = tk.Button(mycanvas, text ="Sử dụng Phương Thức Khác", command = click_button_intput_pass, height=5, width=30)
    Button_intput_pass.place(x = 10,y = 50)

    Button_setup = tk.Button(mycanvas, text ="Cài Đặt", command = helloCallBack, height=5, width=30)
    Button_setup.place(x = 10,y = 170)


    lbl = tk.Label(mycanvas, text="Hình Ảnh", font=("Arial Bold", 15))
    #Xác định vị trí của label
    lbl.place(x=315, y=5)



def get_pairings():
    '''for simulation purposes, this simply randomizes the participants'''
    global participants

    # see http://stackoverflow.com/a/23286332/7432
    shuffle(participants)
    return zip(*[iter(participants)]*2)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("My GUI")
    root.geometry("480x320")


    frame = tk.Frame(root)
    frame.pack(side="top", expand=True, fill="both")
    #root.geometry("480x320")
    draw_postion_palm(frame)

    frame_input_pass = tk.Frame(root)
    frame_input_pass.pack(side="top", expand=True, fill="both")


    root.mainloop()
