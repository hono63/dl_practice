import time
import tkinter, tkinter.filedialog, tkinter.messagebox
from tkinter import ttk

class MeasureTime():
    "スコープ内の時間経過を測定するクラス"
    def __init__(self, display=True):
        self.start = time.time()
        self.display = display
    def __del__(self):
        elapsed_time = time.time() - self.start
        if self.display:
            print("elapsed:{0}".format(elapsed_time) + "[sec]")

class ShowProgress():
    "ループの進捗具合を表示する不真面目なクラス"
    MaxNum   = -1
    Counter  = 1
    Interval = 1
    def __init__(self, max_num):
        if max_num != ShowProgress.MaxNum:
            ShowProgress.MaxNum = max_num
            ShowProgress.Counter = 1
            if isinstance(max_num, int):
                ShowProgress.Interval = (max_num // 1000) * 100
            else:
                ShowProgress.Interval = 1000
        if ShowProgress.Interval == 0:
            ShowProgress.Interval = 100
        if ShowProgress.Counter % ShowProgress.Interval == 1:
            print("progress: " + str(ShowProgress.Counter) + "/" + str(ShowProgress.MaxNum))
            ShowProgress.Counter += 1
        elif isinstance(max_num, int) and ShowProgress.Counter == ShowProgress.MaxNum:
            print("progress: " + str(ShowProgress.Counter) + "/" + str(ShowProgress.MaxNum))
            ShowProgress.reset()
        else:
            ShowProgress.Counter += 1
    @classmethod
    def reset(cls):
        cls.MaxNum   = -1
        cls.Counter  = 1
        cls.Interval = 1

class SelectorGUI():
    "簡単な選択肢ボックス"
    SELECTED = ""
    def __init__(self, value_list, title=""):
        self.root = tkinter.Tk()
        self.root.attributes("-topmost", True)
        self.root.geometry("320x100")
        self.root.title(title)
        self.combo = ttk.Combobox(self.root, state="readonly")
        self.combo["values"] = value_list
        self.combo.current(0)
        self.combo.pack(padx=5, pady=5)
        self.button = tkinter.Button(text="OK", command=self.OK_clicked)
        self.button.pack(padx=5, pady=5)
        self.root.mainloop()
    def OK_clicked(self):
        SelectorGUI.SELECTED = self.combo.get()
        self.root.destroy()

class InputTextGUI():
    "簡単な文字入力ボックス"
    def __init__(self, title=""):
        self.TEXT = ""
        self.root = tkinter.Tk()
        self.root.attributes("-topmost", True)
        self.root.geometry("320x100")
        self.root.title(title)
        self.txt = tkinter.Entry()
        self.txt.pack(padx=5, pady=5)
        self.button = tkinter.Button(text="OK", command=self.OK_clicked)
        self.button.pack(padx=5, pady=5)
        self.root.mainloop()
    def OK_clicked(self):
        self.TEXT = self.txt.get()
        self.root.destroy()

def ShowMessage(text, title="info"):
    "情報ボックス表示"
    root = tkinter.Tk()
    root.withdraw() # 小さなウィンドウを表示させない
    tkinter.messagebox.showinfo(title, text)

def OpenFileGUI(typ=[], save=False, **kwargs):
    "typは [('テキストファイル', '*.txt'), ...] というように指定する"
    selected = ""
    root = tkinter.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    if typ == []: # 空リストの場合はフォルダopen
        selected = tkinter.filedialog.askdirectory(**kwargs)
    elif save: # 保存
        selected = tkinter.filedialog.asksaveasfilename(filetypes=typ, **kwargs)
    else:  # 開く
        selected = tkinter.filedialog.askopenfilename(filetypes=typ, **kwargs)
    root.destroy()
    root.quit()
    return selected
