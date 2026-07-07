import tkinter as tk
from gui import FactorModelApp

def main():
    root = tk.Tk()
    app = FactorModelApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()