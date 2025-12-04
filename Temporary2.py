from tkinter import *
import numpy as np
import cv2
from PIL import ImageTk, Image
from Train_Model import *
import os

root = Tk()
root.title("Facial Attendance System")
flag_webcam = True
success = True
flag_back = False
flag_capture = False
webcam = cv2.VideoCapture(1)
known = r'D:\pythonProject1\face-attendance-system\Known'
flag_continue = False
i = 0

# frame:
frame = LabelFrame(root, text="Face_Recognition", bg="White")

img = Image.open("mobile_screen.png")
img = img.resize((500, 300))
img = ImageTk.PhotoImage(img)
label_image = Label(frame, image=img)


def forget_signup():
    button_capture.grid_forget()
    button_back.grid_forget()
    box.grid_forget()


def back():
    global flag_back
    flag_back = True
def capture():
    global webcam, flag_capture
    flag_capture = True
    webcam.release()
    cv2.destroyAllWindows()

def forget_login():
    button_continue.grid_forget()
    box.grid_forget()


def Continue():
    global flag_continue
    flag_continue = True


def signup():
    global webcam, flag_capture, flag_back, flag_webcam
    if not flag_webcam:
        webcam = cv2.VideoCapture(1)
        flag_webcam = True

    # Forgetting
    button_signup.grid_forget()
    button_login.grid_forget()

    # showing widgets
    box.grid(row=1, column=0)
    button_capture.grid(row=2, column=0)
    button_back.grid(row=3, column=0)
    button_exit.grid(row=4, column=0)

    ret, image = webcam.read()
    if ret:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (500, 300))
        label_image.realone = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_ = ImageTk.PhotoImage(Image.fromarray(img))
        label_image.image = img_
        label_image.configure(image=label_image.image)
        label_image.grid(row=0, column=0)

    if flag_back:
        webcam.release()
        cv2.destroyAllWindows()
        flag_webcam = False
        forget_signup()
        main()
        flag_back = False
        if flag_capture:
            forget_signup()
            main()
            flag_capture = False
            person = box.get('1.0', 'end-1c')
            print(person)
            known_people = os.listdir(known)
            if person not in known_people:
                os.mkdir(known + '/' + person)
                i = len(os.listdir(known + '/' + person))
                folder = 'known/' + person
                cv2.imwrite(folder + '/' + f'{person}{i + 1}.jpg', label_image.realone)
                print("Person Saved.")
                label_image.image = None
                train_model()
            else:
                label_image.image = None
                print("Already Signedup.Just go to Login")
            box.delete('1.0', 'end')
        elif not flag_capture:
            label_image.image = None
            label_image.configure(image=label_image.image)
            flag_capture = False
            flag_webcam = False

        return
    if flag_back:
        webcam.release()
        cv2.destroyAllWindows()
        flag_webcam = False
        forget_signup()
        main()
        flag_back = False
    label_image.after(10, signup)


def login():
    known_embeddings = np.load("known_embeddings.npy")
    known_names = np.load("known_names.npy")
    global webcam, flag_capture, flag_back, flag_webcam,flag_continue
    forget_main()

    if not flag_webcam:
        webcam = cv2.VideoCapture(1)
        flag_webcam = True

    # widgets to show
    box.grid(row=1, column=0)
    button_continue.grid(row=2, column=0)
    button_back.grid(row=3, column=0)
    button_exit.grid(row=4, column=0)
    ret, image = webcam.read()
    if ret:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (500, 300))
        label_image.realone = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        label_image.image = img
        label_image.configure(image=label_image.image)
        label_image.grid(row=0, column=0)
    if flag_back:
        button_continue.grid_forget()
        label_image.image = None
        webcam.release()
        cv2.destroyAllWindows()
        flag_webcam = False
        forget_signup()
        main()
        flag_back = False
        return
    face_name = box.get('1.0', 'end-1c')

    # Face Detection and Recognition:
    if face_name in known_names:
        if flag_continue:
            best_distance = 999
            label_image.faceimage, x_start, y_start, height, width = detect_and_crop(label_image.realone)
            if label_image.faceimage is None:
                IMage = Image.open(r'no_face_detected.png')
                IMage = IMage.resize((500, 300))
                IMage = ImageTk.PhotoImage(IMage)
                label_image.image = IMage  # store reference to avoid garbage collection
                label_image.configure(image=label_image.image)
                forget_login()
                if flag_back:
                    flag_back = False
                    flag_continue = False
                    label_image.image = None
                    main()
                    return
            embedding = embeddings(label_image.faceimage)
            for index, i in enumerate(known_embeddings):
                dist = np.linalg.norm(embedding - i)
                if abs(dist) < 0.9:
                    IMage = Image.open(r'LoginSuccess.png')
                    IMage = IMage.resize((500, 300))
                    IMage = ImageTk.PhotoImage(IMage)
                    label_image.image = IMage  # store reference to avoid garbage collection
                    label_image.configure(image=label_image.image)
                    print("Congratulation,You have just Logged in!")
                    forget_login()
                    if flag_back:
                        main()
                        button_back.grid_forget()
                        flag_back = False
                        flag_continue = False
                        return
                else:
                    IMage = Image.open(r'loginUnsuccessful.png')
                    IMage = IMage.resize((500, 300))
                    IMage = ImageTk.PhotoImage(IMage)
                    label_image.image = IMage  # store reference to avoid garbage collection
                    label_image.configure(image=label_image.image)
                    forget_login()
                    button_back.grid_forget()
                    return

    flag_continue = False
    label_image.after(10, login)


def main():
    buttons()
    faceframe()


def forget_main():
    button_signup.grid_forget()
    button_login.grid_forget()


# Objects
button_signup = Button(root, text="  Sign Up", font=("Arial", 20, "bold"), command=signup)
button_login = Button(root, text="  Login  ", font=("Arial", 20, "bold"), command=login)
button_exit = Button(root, text="    Exit  ", font=("Arial", 20, "bold"), command=root.quit)
button_capture = Button(root, text="Capture", font=("Arial", 20, "bold"), command=capture)
button_back = Button(root, text="   Back  ", font=("Arial", 20, "bold"), command=back)
button_continue = Button(root, text="Continue", font=("Arial", 20, "bold"), command=Continue)
box = Text(root, height=2, width=50)
# Column Labels:
label4 = Label(root, text="                        ")
label5 = Label(root, text="                        ")
label6 = Label(root, text="                        ")
label7 = Label(root, text="                        ")


# functions
def buttons():
    button_signup.grid(row=0, column=10, sticky='n')
    button_login.grid(row=0, column=10)
    button_exit.grid(row=0, column=10, sticky='s')


def faceframe():
    frame.grid(row=0, column=0)
    label4.grid(row=0, column=2)
    label5.grid(row=1, column=2)
    label6.grid(row=2, column=2)
    label_image.grid()


main()
root.mainloop()
