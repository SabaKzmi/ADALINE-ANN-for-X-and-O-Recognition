from tkinter import messagebox
import customtkinter as ctk
import pandas as pd
pd.options.mode.chained_assignment = None

from Adaline import Adaline

class xoGUI:
    def __init__(self):
        self.learningDF = None
        self.CheckBoxData = []
        #window
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)

        self.root = ctk.CTk()

        self.root.geometry("800x400")
        self.root.title("XO Adaline")

        tabs = ctk.CTkTabview(self.root,width=790, height=390, corner_radius=10,)
        tabs.pack(pady=10)

        # Create tabs
        self.learningTab = tabs.add("Make Data")
        self.testTab = tabs.add("Test")

        #label: X or O?
        self.label = ctk.CTkLabel(self.learningTab, text="Is it an X or an O?", font = ("Helvetica", 20))
        self.label.place(x = 490, y = 80)

        #submit button
        self.btnInput = ctk.CTkButton(self.learningTab, text = "submit", font = ('Arial', 15), command = self.submit)
        self.btnInput.place(x = 500, y = 230)

        #radio button xo
        self.radioVar = ctk.IntVar(value = 3)
        self.radioX = ctk.CTkRadioButton(self.learningTab, text = "X", font = ('Arial', 15), value = 1, variable = self.radioVar)
        self.radioX.place(x = 500, y = 130)
        self.radioY = ctk.CTkRadioButton(self.learningTab, text = "O", font = ('Arial', 15), value = -1, variable = self.radioVar)
        self.radioY.place(x = 500, y = 180)

        # it's sth label
        self.itsLabel = ctk.CTkLabel(self.testTab, text="It's an ", font=("Helvetica", 20))
        self.itsLabel.place(x=490, y=150)

        # result label
        self.resultLabel = ctk.CTkLabel(self.testTab, text=" ", font=("Helvetica", 50))
        self.resultLabel.place(x=585, y=165)

        # test button
        self.btnTest = ctk.CTkButton(self.testTab, text="Test", font=('Arial', 15), command=self.test)
        self.btnTest.place(x=500, y=260)

        # clear test button
        self.btnTest = ctk.CTkButton(self.testTab, text="Clear", font=('Arial', 15), command=self.cleanT, height=28, width=25)
        self.btnTest.place(x=645, y=260)

        # Train button
        self.btnTest = ctk.CTkButton(self.testTab, text="Train", font=('Arial', 15), command=self.train, height=28, width=200)
        self.btnTest.place(x=490, y=60)

        #button frame - make data grid
        self.col0Frame = ctk.CTkFrame(self.learningTab)
        self.col0Frame.rowconfigure(0, weight = 1)
        self.col0Frame.rowconfigure(1, weight = 1)
        self.col0Frame.rowconfigure(2, weight = 1)
        self.col0Frame.rowconfigure(3, weight = 1)
        self.col0Frame.rowconfigure(4, weight = 1)

        self.col1Frame = ctk.CTkFrame(self.learningTab)
        self.col1Frame.rowconfigure(0, weight=1)
        self.col1Frame.rowconfigure(1, weight=1)
        self.col1Frame.rowconfigure(2, weight=1)
        self.col1Frame.rowconfigure(3, weight=1)
        self.col1Frame.rowconfigure(4, weight=1)

        self.col2Frame = ctk.CTkFrame(self.learningTab)
        self.col2Frame.rowconfigure(0, weight=1)
        self.col2Frame.rowconfigure(1, weight=1)
        self.col2Frame.rowconfigure(2, weight=1)
        self.col2Frame.rowconfigure(3, weight=1)
        self.col2Frame.rowconfigure(4, weight=1)

        self.col3Frame = ctk.CTkFrame(self.learningTab)
        self.col3Frame.rowconfigure(0, weight=1)
        self.col3Frame.rowconfigure(1, weight=1)
        self.col3Frame.rowconfigure(2, weight=1)
        self.col3Frame.rowconfigure(3, weight=1)
        self.col3Frame.rowconfigure(4, weight=1)

        self.col4Frame = ctk.CTkFrame(self.learningTab)
        self.col4Frame.rowconfigure(0, weight=1)
        self.col4Frame.rowconfigure(1, weight=1)
        self.col4Frame.rowconfigure(2, weight=1)
        self.col4Frame.rowconfigure(3, weight=1)
        self.col4Frame.rowconfigure(4, weight=1)

        # col 0
        self.btn00 = ctk.CTkCheckBox(self.col0Frame, text="", onvalue = 1, offvalue = -1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color = '#2FA572', corner_radius = 9)
        self.btn00.grid(row =0, column =0, sticky = "ew", padx = 5, pady = 5)
        self.btn10 = ctk.CTkCheckBox(self.col0Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color = '#2FA572', corner_radius = 9)
        self.btn10.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.btn20 = ctk.CTkCheckBox(self.col0Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color = '#2FA572', corner_radius = 9)
        self.btn20.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.btn30 = ctk.CTkCheckBox(self.col0Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color = '#2FA572', corner_radius = 9)
        self.btn30.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.btn40 = ctk.CTkCheckBox(self.col0Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn40.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        # column 1
        self.btn01 = ctk.CTkCheckBox(self.col1Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn01.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.btn11 = ctk.CTkCheckBox(self.col1Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn11.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.btn21 = ctk.CTkCheckBox(self.col1Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn21.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        self.btn31 = ctk.CTkCheckBox(self.col1Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn31.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        self.btn41 = ctk.CTkCheckBox(self.col1Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn41.grid(row=4, column=1, sticky="ew", padx=5, pady=5)

        # col 2
        self.btn02 = ctk.CTkCheckBox(self.col2Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn02.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.btn12 = ctk.CTkCheckBox(self.col2Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn12.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.btn22 = ctk.CTkCheckBox(self.col2Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn22.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.btn32 = ctk.CTkCheckBox(self.col2Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn32.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.btn42 = ctk.CTkCheckBox(self.col2Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn42.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        # col 3
        self.btn03 = ctk.CTkCheckBox(self.col3Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn03.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.btn13 = ctk.CTkCheckBox(self.col3Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn13.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.btn23 = ctk.CTkCheckBox(self.col3Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn23.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.btn33 = ctk.CTkCheckBox(self.col3Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn33.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.btn43 = ctk.CTkCheckBox(self.col3Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn43.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        # col 4
        self.btn04 = ctk.CTkCheckBox(self.col4Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn04.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.btn14 = ctk.CTkCheckBox(self.col4Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn14.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.btn24 = ctk.CTkCheckBox(self.col4Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn24.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.btn34 = ctk.CTkCheckBox(self.col4Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn34.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.btn44 = ctk.CTkCheckBox(self.col4Frame, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn44.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        self.col0Frame.place(x=70, y=10)
        self.col1Frame.place(x=125, y=10)
        self.col2Frame.place(x=180, y=10)
        self.col3Frame.place(x=235, y=10)
        self.col4Frame.place(x=290, y=10)

        # button frame - test grid
        self.col0FrameT = ctk.CTkFrame(self.testTab)
        self.col0FrameT.rowconfigure(0, weight=1)
        self.col0FrameT.rowconfigure(1, weight=1)
        self.col0FrameT.rowconfigure(2, weight=1)
        self.col0FrameT.rowconfigure(3, weight=1)
        self.col0FrameT.rowconfigure(4, weight=1)

        self.col1FrameT = ctk.CTkFrame(self.testTab)
        self.col1FrameT.rowconfigure(0, weight=1)
        self.col1FrameT.rowconfigure(1, weight=1)
        self.col1FrameT.rowconfigure(2, weight=1)
        self.col1FrameT.rowconfigure(3, weight=1)
        self.col1FrameT.rowconfigure(4, weight=1)

        self.col2FrameT = ctk.CTkFrame(self.testTab)
        self.col2FrameT.rowconfigure(0, weight=1)
        self.col2FrameT.rowconfigure(1, weight=1)
        self.col2FrameT.rowconfigure(2, weight=1)
        self.col2FrameT.rowconfigure(3, weight=1)
        self.col2FrameT.rowconfigure(4, weight=1)

        self.col3FrameT = ctk.CTkFrame(self.testTab)
        self.col3FrameT.rowconfigure(0, weight=1)
        self.col3FrameT.rowconfigure(1, weight=1)
        self.col3FrameT.rowconfigure(2, weight=1)
        self.col3FrameT.rowconfigure(3, weight=1)
        self.col3FrameT.rowconfigure(4, weight=1)

        self.col4FrameT = ctk.CTkFrame(self.testTab)
        self.col4FrameT.rowconfigure(0, weight=1)
        self.col4FrameT.rowconfigure(1, weight=1)
        self.col4FrameT.rowconfigure(2, weight=1)
        self.col4FrameT.rowconfigure(3, weight=1)
        self.col4FrameT.rowconfigure(4, weight=1)

        # col 0 T
        self.btn00T = ctk.CTkCheckBox(self.col0FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn00T.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.btn10T = ctk.CTkCheckBox(self.col0FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn10T.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.btn20T = ctk.CTkCheckBox(self.col0FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn20T.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.btn30T = ctk.CTkCheckBox(self.col0FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn30T.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.btn40T = ctk.CTkCheckBox(self.col0FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn40T.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        # column 1 T
        self.btn01T = ctk.CTkCheckBox(self.col1FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn01T.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.btn11T = ctk.CTkCheckBox(self.col1FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn11T.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.btn21T = ctk.CTkCheckBox(self.col1FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn21T.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        self.btn31T = ctk.CTkCheckBox(self.col1FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn31T.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        self.btn41T = ctk.CTkCheckBox(self.col1FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn41T.grid(row=4, column=1, sticky="ew", padx=5, pady=5)

        # col 2 T
        self.btn02T = ctk.CTkCheckBox(self.col2FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn02T.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.btn12T = ctk.CTkCheckBox(self.col2FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn12T.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.btn22T = ctk.CTkCheckBox(self.col2FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn22T.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.btn32T = ctk.CTkCheckBox(self.col2FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn32T.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.btn42T = ctk.CTkCheckBox(self.col2FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn42T.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        # col 3 T
        self.btn03T = ctk.CTkCheckBox(self.col3FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn03T.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.btn13T = ctk.CTkCheckBox(self.col3FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn13T.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.btn23T = ctk.CTkCheckBox(self.col3FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn23T.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.btn33T = ctk.CTkCheckBox(self.col3FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn33T.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.btn43T = ctk.CTkCheckBox(self.col3FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn43T.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        # col 4 T
        self.btn04T = ctk.CTkCheckBox(self.col4FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn04T.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.btn14T = ctk.CTkCheckBox(self.col4FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn14T.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.btn24T = ctk.CTkCheckBox(self.col4FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn24T.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.btn34T = ctk.CTkCheckBox(self.col4FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn34T.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.btn44T = ctk.CTkCheckBox(self.col4FrameT, text="", onvalue=1, offvalue=-1,
                                     checkbox_width=50, checkbox_height=50, checkmark_color='#2FA572', corner_radius=9)
        self.btn44T.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        self.col0FrameT.place(x=70, y=10)
        self.col1FrameT.place(x=125, y=10)
        self.col2FrameT.place(x=180, y=10)
        self.col3FrameT.place(x=235, y=10)
        self.col4FrameT.place(x=290, y=10)

        self.root.mainloop()

    def submit(self):
        if self.radioVar.get() == 3:
             messagebox.showerror("Error", "Please select an option")
        else:
             self.getCheckBoxes()
             self.clearCheckBoxes()

    def test(self):
        adaline = Adaline()
        res = adaline.test(self.getCheckBoxesTest())
        self.resultLabel.configure(text=res)

    def train(self):
            adaline = Adaline()
            adaline.train()

    def getCheckBoxes(self):

        CheckBoxData = [self.btn00.get(), self.btn01.get(), self.btn02.get(), self.btn03.get(), self.btn04.get(),
                             self.btn10.get(), self.btn11.get(), self.btn12.get(), self.btn13.get(), self.btn14.get(),
                             self.btn20.get(), self.btn21.get(), self.btn22.get(), self.btn23.get(), self.btn24.get(),
                             self.btn30.get(), self.btn31.get(), self.btn32.get(), self.btn33.get(), self.btn34.get(),
                             self.btn40.get(), self.btn41.get(), self.btn42.get(), self.btn43.get(), self.btn44.get(),
                             self.radioVar.get(), -1 * self.radioVar.get()]

        columnNames = ['x00', 'x01', 'x02', 'x03', 'x04',
                       'x10', 'x11', 'x12', 'x13', 'x14',
                       'x20', 'x21', 'x22', 'x23', 'x24',
                       'x30', 'x31', 'x32', 'x33', 'x34',
                       'x40', 'x41', 'x42', 'x43', 'x44', 't1', 't2']

        newRow = {columnNames[i] : CheckBoxData[i] for i in range(0, len(columnNames))}

        self.learningDF = pd.read_excel('XOdata.xlsx')
        self.learningDF = self.learningDF._append(newRow, ignore_index=True)
        self.learningDF.drop_duplicates(keep = 'first')
        self.learningDF.to_excel('XOdata.xlsx', index=False)

    def clearCheckBoxes(self):
        self.btn00.deselect()
        self.btn01.deselect()
        self.btn02.deselect()
        self.btn03.deselect()
        self.btn04.deselect()

        self.btn10.deselect()
        self.btn11.deselect()
        self.btn12.deselect()
        self.btn13.deselect()
        self.btn14.deselect()

        self.btn20.deselect()
        self.btn21.deselect()
        self.btn22.deselect()
        self.btn23.deselect()
        self.btn24.deselect()

        self.btn30.deselect()
        self.btn31.deselect()
        self.btn32.deselect()
        self.btn33.deselect()
        self.btn34.deselect()

        self.btn40.deselect()
        self.btn41.deselect()
        self.btn42.deselect()
        self.btn43.deselect()
        self.btn44.deselect()

        self.radioVar.set(3)

    def getCheckBoxesTest(self):
        CheckBoxT = [self.btn00T.get(), self.btn01T.get(), self.btn02T.get(), self.btn03T.get(), self.btn04T.get(),
                     self.btn10T.get(), self.btn11T.get(), self.btn12T.get(), self.btn13T.get(), self.btn14T.get(),
                     self.btn20T.get(), self.btn21T.get(), self.btn22T.get(), self.btn23T.get(), self.btn24T.get(),
                     self.btn30T.get(), self.btn31T.get(), self.btn32T.get(), self.btn33T.get(), self.btn34T.get(),
                     self.btn40T.get(), self.btn41T.get(), self.btn42T.get(), self.btn43T.get(), self.btn44T.get()]
        return CheckBoxT

    def cleanT(self):
        self.btn00T.deselect()
        self.btn01T.deselect()
        self.btn02T.deselect()
        self.btn03T.deselect()
        self.btn04T.deselect()

        self.btn10T.deselect()
        self.btn11T.deselect()
        self.btn12T.deselect()
        self.btn13T.deselect()
        self.btn14T.deselect()

        self.btn20T.deselect()
        self.btn21T.deselect()
        self.btn22T.deselect()
        self.btn23T.deselect()
        self.btn24T.deselect()

        self.btn30T.deselect()
        self.btn31T.deselect()
        self.btn32T.deselect()
        self.btn33T.deselect()
        self.btn34T.deselect()

        self.btn40T.deselect()
        self.btn41T.deselect()
        self.btn42T.deselect()
        self.btn43T.deselect()
        self.btn44T.deselect()

        self.resultLabel.configure(text=' ')
