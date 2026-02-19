import sys
import tkinter as tk
from tkinter import messagebox
print (f"Python version : {sys.version}")
def on_button_click():
  user_name = entry.get()
  if user_name:
    messagebox.showinfo("Info", f"Welcome {user_name}!")
  else:
    messagebox.showwarning("Warning", "Please input user name")
def on_exit():
  if messsagebox.askquestion("Question", "Are you sure to exit?") == 'yes':
    root.quit()

root = tk.Tk()
root.title("My first Tkinter Example")
root.geometry("450x350")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (450 // 2)
y = (screen_height // 2) - (350 // 2)
root.geometry(f"450x350+{x}+{y}")
root.mainloop()
