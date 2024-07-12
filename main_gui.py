from tkinter import filedialog
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
import numpy as np
from em_image import em_image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from skimage import io
import threading

instructions = '''
1. Use the Open  File button to open and image file.
2. Adjust the Wedge size and Theta values to define the wedge Use the View wedge button to make sure that the size and angle of the wedge covers the noise in the Fourier space.
3. Use the add wedge button to add the current values to the processing parameters.
4. You can view and edit the already added wedges using the provided drop down button.
5. Once satisfied with the parameters click on remove stripe button to strat the processing. Once the processing is complete the button should turn green.
6. Use the show processed Button to view the processed image and the fft.
7. If you encounter a significant loss in contrast adjust the p1 and p2 values perform a simple contrast enhancement.

App created by Debadutta Patra
'''

class settings_frame(tb.Frame):
    def __init__(self, master, current_theme,):

        super().__init__(master, bootstyle='secondary')
        self.style = tb.Style()
        self.style.configure('TButton', font=("Helvetica", 14))

        self.available_themes = self.master.style.theme_names()
        self.progress_bar = tb.Progressbar(
            self, bootstyle='sucess striped', mode=INDETERMINATE, length=400)
        self.progress_bar.grid(column=0, row=0, columnspan=2,
                               padx=10, pady=10, sticky=NSEW)

        self.theme_box = tb.Combobox(self,
                                     bootstyle='info', values=self.available_themes, font=("Helvetica", 14))
        self.theme_box.grid(column=0, row=1, padx=10, pady=10, sticky='nsew')
        self.theme_box.current(self.available_themes.index(current_theme))
        self.theme_box.bind("<<ComboboxSelected>>", self.click_bind)
        self.quit_button = tb.Button(self,
                                     text='Quit', bootstyle='danger', command=self.quit_app)
        self.quit_button.grid(column=1, row=1, padx=10, pady=10, sticky='nsew')

    def click_bind(self, e):
        self.master.style.theme_use(self.theme_box.get())

    def quit_app(self):
        self.master.quit()


class image_tabs(tb.Notebook):
    def __init__(self, master):
        super().__init__(master, bootstyle='info', height=700, width=700)
        self.style = tb.Style()
        self.style.configure('TNotebook.Tab', font=("Helvetica", 14))
        self.in_img = tb.Frame(self)
        self.in_fft = tb.Frame(self)
        self.out_img = tb.Frame(self)
        self.out_fft = tb.Frame(self)
        self.enh_img = tb.Frame(self)
        self.instructions = tb.Frame(self)
        self.instructions_box = tb.Text(self.instructions,wrap=WORD,font=('Helvetica',20))
        self.instructions_box.pack(fill=BOTH,expand=YES)
        self.instructions_box.insert(END,instructions)
        self.instructions_box.config(state=DISABLED)
        self.add(self.in_img, text='Input Image')
        self.add(self.in_fft, text='Input FFT')
        self.add(self.out_img, text='Recon Image')
        self.add(self.out_fft, text='Recon FFT')
        self.add(self.enh_img, text='Enhanced Image')
        self.add(self.instructions, text='Instructions')


class io_frame(tb.Frame):
    def __init__(self, master):
        super().__init__(master, bootstyle='Secondary')
        self.style = tb.Style()
        self.style.configure('TButton', font=("Helvetica", 14))

#       button to open input file
        self.input_button = tb.Button(
            self, text='Open file', bootstyle='primary', command=self.master.initialize)
        self.input_button.grid(column=0, row=0, padx=10,
                               pady=10, columnspan=2, sticky='nsew')

#       widgets for setting the wedge size
#       wedge size
        self.wedge_label = tb.Label(
            self, text='Wedge Size:', bootstyle='primary', font=('Helvetica', 14))
        self.wedge_label.grid(column=0, row=1, padx=10, pady=10, sticky='e')
        self.wedge_spinbox = tb.Spinbox(self,
                                        from_=0, to=180, increment=0.2, bootstyle='primary')
        self.wedge_spinbox.set(0)
        self.wedge_spinbox.grid(column=1, row=1, padx=10, pady=10)

#       angle theta
        self.theta_label = tb.Label(
            self, text='Theta:', bootstyle='primary', font=('Helvetica', 14))
        self.theta_label.grid(column=0, row=2, padx=10, pady=10, sticky='e')
        self.theta_spinbox = tb.Spinbox(self,
                                        from_=-90, to=90, increment=0.2, bootstyle='primary')
        self.theta_spinbox.set(0)
        self.theta_spinbox.grid(column=1, row=2, padx=10, pady=10)

        self.kmin_label = tb.Label(
            self, text='K_min:', bootstyle='primary', font=('Helvetica', 14))
        self.kmin_label.grid(column=0, row=3, padx=10, pady=10, sticky='e')
        self.kmin_spinbox = tb.Spinbox(self,
                                       from_=1, to=200, increment=1, bootstyle='primary')
        self.kmin_spinbox.set(15)
        self.kmin_spinbox.grid(column=1, row=3, padx=10, pady=10)

        self.wedge_list = tb.Combobox(self, bootstyle='info', values=[])
        self.wedge_list.grid(column=0, row=4, padx=10, pady=10,)
        self.wedge_list.bind('<<ComboboxSelected>>', self.master.get_wedge)

        self.add_wedge_button = tb.Button(
            self, text="Add Wedge", bootstyle='primary', command=self.master.add_wedge)
        self.add_wedge_button.grid(
            column=1, row=4, padx=10, pady=10, sticky='nsew')
        self.add_wedge_button.config(state=DISABLED)

        self.update_wedge_button = tb.Button(
            self, text="Update Wedge", bootstyle='primary', command=self.master.edit_wedge)
        self.update_wedge_button.grid(
            column=1, row=5, padx=10, pady=10, sticky='nsew')
        self.update_wedge_button.config(state=DISABLED)

        self.delete_wedge_button = tb.Button(
            self, text="Delete Wedge", bootstyle='warning', command=self.master.delete_wedge)
        self.delete_wedge_button.grid(
            column=0, row=5, padx=10, pady=10, sticky='nsew')
        self.delete_wedge_button.config(state=DISABLED)

        self.delete_all_wedge_button = tb.Button(
            self, text='Delete all wedges', bootstyle=DANGER, command=self.master.delete_all_wedges)
        self.delete_all_wedge_button.config(state=DISABLED)
        self.delete_all_wedge_button.grid(
            column=0, row=6, padx=10, pady=10, sticky='nsew')

        self.view_wedge_button = tb.Button(
            self, text="View Wedge", bootstyle='primary', command=self.master.view_wedge)
        self.view_wedge_button.grid(
            column=1, row=6, padx=10, pady=10, sticky='nsew')
        self.view_wedge_button.config(state=DISABLED)

        self.process_image_button = tb.Button(
            self, text="Remove Stripes", bootstyle='primary', command=self.master.process_image)
        self.process_image_button.grid(
            column=0, row=7, padx=10, pady=10, sticky='nsew')
        self.process_image_button.config(state=DISABLED)

        self.view_processed_image_button = tb.Button(
            self, text='Show Processed', bootstyle=PRIMARY, command=self.master.view_processed)
        self.view_processed_image_button.grid(
            column=1, row=7, padx=10, pady=10, sticky=NSEW)
        self.view_processed_image_button.config(state=DISABLED)

        self.save_image = tb.Button(
            self, text='Save Processed Image', bootstyle='primary', command=self.master.save_processed)
        self.save_image.grid(column=0, row=8, padx=10,
                             pady=10, sticky='nsew', columnspan=2)
        self.save_image.config(state=DISABLED)

        self.p1_label = tb.Label(
            self, text='p_1:', bootstyle='primary', font=('Helvetica', 14))
        self.p1_label.grid(column=0, row=9, padx=10, pady=10, sticky='e')

        self.p1_spinbox = tb.Spinbox(self,
                                     from_=0, to=100, increment=0.1, bootstyle='primary')
        self.p1_spinbox.set(1)
        self.p1_spinbox.grid(column=1, row=9, padx=10, pady=10)

        self.p2_label = tb.Label(
            self, text='p_2:', bootstyle='primary', font=('Helvetica', 14))
        self.p2_label.grid(column=0, row=10, padx=10, pady=10, sticky='e')
        self.p2_spinbox = tb.Spinbox(self,
                                     from_=0, to=100, increment=0.1, bootstyle='primary')
        self.p2_spinbox.set(90)
        self.p2_spinbox.grid(column=1, row=10, padx=10, pady=10)

        self.enhance_button = tb.Button(
            self, text='Enhance Image', bootstyle='primary', command=self.master.adjust_contrast)
        self.enhance_button.grid(
            column=1, row=11, padx=10, pady=10, sticky='nsew')
        self.enhance_button.config(state=DISABLED)

        self.save_enhanced_button = tb.Button(
            self, text='Save Enhanced Image', bootstyle='primary', command=self.master.save_enhanced)
        self.save_enhanced_button.grid(
            column=0, row=12, padx=10, pady=10, columnspan=2, sticky='nsew')
        self.save_enhanced_button.config(state=DISABLED)


class MyApp(tb.Window):
    def __init__(self, title="Stripe Begone", themename="litera", iconphoto='./icon.png', size=None,
                 position=None, minsize=None, maxsize=None, resizable=None, hdpi=True, scaling=None,
                 transient=None, overrideredirect=False, alpha=1):
        super().__init__(title, themename, iconphoto, size, position, minsize,
                         maxsize, resizable, hdpi, scaling, transient, overrideredirect, alpha)
        self.title(title)
        self.geometry('1200x900')
        self.process_frame = io_frame(self)
        self.process_frame.grid(
            row=0, column=0, padx=10, pady=10, sticky='nsew')

        self.setting_frame = settings_frame(
            self, themename)
        self.setting_frame.grid(
            row=1, column=0, padx=10, pady=10, sticky='nsew')

        self.img_panels = image_tabs(self)
        self.img_panels.grid(row=0, column=1, rowspan=2,
                             padx=10, pady=10, sticky='nsew')

        self.canvas, self.toolbar, self.canvas_fft, self.toolbar_fft = None, None, None, None
        self.canvas_out, self.toolbar_out, self.canvas_fft_out, self.toolbar_fft_out = None, None, None, None
        self.canvas_enh, self.toolbar_enh = None, None

    def show_input(self):
        fig, axis = plt.subplots(1, 1, figsize=(8, 8))
        axis.imshow(self.em_file.image, cmap='gray')
        axis.set_title('Input Image')
        axis.axis('off')
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
            self.toolbar.forget()
        self.canvas = FigureCanvasTkAgg(fig, master=self.img_panels.in_img)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas,
                                            self.img_panels.in_img)
        self.toolbar.update()
        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().pack()

    def show_input_fft(self):
        fig, axis = plt.subplots(1, 1, figsize=(8, 8))
        axis.imshow(np.log(np.abs(self.em_file.fft_shifted)), cmap='gray')
        axis.set_title('Input Image FFT')
        axis.axis('off')
        if self.canvas_fft:
            self.canvas_fft.get_tk_widget().pack_forget()
            self.toolbar_fft.forget()
        self.canvas_fft = FigureCanvasTkAgg(fig, master=self.img_panels.in_fft)
        self.canvas_fft.draw()
        self.toolbar_fft = NavigationToolbar2Tk(self.canvas_fft,
                                                self.img_panels.in_fft)
        self.toolbar_fft.update()
        # placing the toolbar on the Tkinter window
        self.canvas_fft.get_tk_widget().pack()

    def show_wedges(self, mask):
        fig, axis = plt.subplots(1, 1, figsize=(8, 8))
        axis.imshow(np.log(np.abs(self.em_file.fft_shifted)), cmap='gray')
        axis.imshow(mask, cmap='RdBu', alpha=0.1)
        axis.set_title('Input Image FFT')
        axis.axis('off')
        if self.canvas_fft:
            self.canvas_fft.get_tk_widget().pack_forget()
            self.toolbar_fft.forget()
        self.canvas_fft = FigureCanvasTkAgg(fig, master=self.img_panels.in_fft)
        self.canvas_fft.draw()
        self.toolbar_fft = NavigationToolbar2Tk(self.canvas_fft,
                                                self.img_panels.in_fft)
        self.toolbar_fft.update()
        # placing the toolbar on the Tkinter window
        self.canvas_fft.get_tk_widget().pack()

    def show_output(self):
        fig, axis = plt.subplots(1, 1, figsize=(8, 8))
        axis.imshow(self.em_file.img_recon, cmap='gray')
        axis.set_title('Reconstructed Image')
        axis.axis('off')
        if self.canvas_out:
            self.canvas_out.get_tk_widget().pack_forget()
            self.toolbar_out.forget()
        self.canvas_out = FigureCanvasTkAgg(
            fig, master=self.img_panels.out_img)
        self.canvas_out.draw()
        self.toolbar_out = NavigationToolbar2Tk(self.canvas_out,
                                                self.img_panels.out_img)
        self.toolbar_out.update()
        # placing the toolbar on the Tkinter window
        self.canvas_out.get_tk_widget().pack()

    def show_output_fft(self):
        fig, axis = plt.subplots(1, 1, figsize=(8, 8))
        axis.imshow(np.log(np.abs(self.em_file.processed_fft)), cmap='gray')
        axis.set_title('Processed FFT')
        axis.axis('off')
        if self.canvas_fft_out:
            self.canvas_fft_out.get_tk_widget().pack_forget()
            self.toolbar_fft_out.forget()
        self.canvas_fft_out = FigureCanvasTkAgg(
            fig, master=self.img_panels.out_fft)
        self.canvas_fft_out.draw()
        self.toolbar_fft_out = NavigationToolbar2Tk(self.canvas_fft_out,
                                                    self.img_panels.out_fft)
        self.toolbar_fft_out.update()
        # placing the toolbar on the Tkinter window
        self.canvas_fft_out.get_tk_widget().pack()

    def show_enhanced(self):
        fig, axis = plt.subplots(1, 1, figsize=(8, 8))
        axis.imshow(self.em_file.img_enhanced, cmap='gray')
        axis.set_title('Reconstructed and Contrast Adjusted Image')
        axis.axis('off')
        if self.canvas_enh:
            self.canvas_enh.get_tk_widget().pack_forget()
            self.toolbar_enh.forget()
        self.canvas_enh = FigureCanvasTkAgg(
            fig, master=self.img_panels.enh_img)
        self.canvas_enh.draw()
        self.toolbar_enh = NavigationToolbar2Tk(self.canvas_enh,
                                                self.img_panels.enh_img)
        self.toolbar_enh.update()
        # placing the toolbar on the Tkinter window
        self.canvas_enh.get_tk_widget().pack()

    def initialize(self):
        try:
            self.em_file = em_image(filedialog.askopenfilename())
            try:
                self.canvas.get_tk_widget().pack_forget()
                self.canvas_fft.get_tk_widget().pack_forget()
                self.canvas_out.get_tk_widget().pack_forget()
                self.canvas_fft_out.get_tk_widget().pack_forget()
                self.canvas_enh.get_tk_widget().pack_forget()
                self.toolbar.forget()
                self.toolbar_fft.forget()
                self.toolbar_out.forget()
                self.toolbar_fft_out.forget()
                self.toolbar_enh.forget()
            except:
                pass
            self.show_input()
            self.show_input_fft()
            self.process_frame.add_wedge_button.config(state=ACTIVE)
            self.process_frame.view_wedge_button.config(state=ACTIVE)
            self.process_frame.wedge_list.configure(values=[])
            self.process_frame.delete_all_wedge_button.config(state=DISABLED)
            self.process_frame.delete_wedge_button.config(state=DISABLED)
            self.process_frame.update_wedge_button.config(state=DISABLED)
            self.process_frame.save_enhanced_button.config(state=DISABLED)
            self.process_frame.view_processed_image_button.config(state=DISABLED)
            self.process_frame.enhance_button.config(state=DISABLED)
            self.process_frame.process_image_button.config(state=DISABLED)
        except:
            io_error = Messagebox.show_error(
                message="Please select a valid image file", title='Invalid file format', alert=True)

    def add_wedge(self):
        self.em_file.add_wedge(np.float64(self.process_frame.wedge_spinbox.get()), np.float64(
            self.process_frame.theta_spinbox.get()), np.float64(self.process_frame.kmin_spinbox.get()))
        self.process_frame.wedge_list.configure(
            values=[f'Wedge {x+1}' for x in range(len(self.em_file.wedge_size))])
        self.process_frame.process_image_button.config(bootstyle=PRIMARY)
        self.process_frame.delete_wedge_button.config(state=ACTIVE)
        self.process_frame.update_wedge_button.config(state=ACTIVE)
        self.process_frame.process_image_button.config(state=ACTIVE)
        self.process_frame.delete_all_wedge_button.config(state=ACTIVE)

    def get_wedge(self, e):
        wedge_index = np.int64(
            self.process_frame.wedge_list.get().split()[-1])-1
        self.process_frame.wedge_spinbox.set(
            self.em_file.wedge_size[wedge_index])
        self.process_frame.theta_spinbox.set(
            self.em_file.wedge_angle[wedge_index])
        self.process_frame.kmin_spinbox.set(self.em_file.k_min[wedge_index])

    def delete_wedge(self):
        if self.process_frame.wedge_list.get() != "":
            delete_warning = Messagebox.yesno(message=f"Do you want to delete '{
                self.process_frame.wedge_list.get()}'", title='Warning', alert=True)
            if delete_warning == "Yes":
                wedge_index = np.int64(
                    self.process_frame.wedge_list.get().split()[-1])-1
                self.em_file.delete_mask(wedge_index)
                self.process_frame.wedge_list.set('')
                self.process_frame.process_image_button.config(
                    bootstyle=PRIMARY)
                if len(self.em_file.wedge_size) > 0:
                    self.process_frame.wedge_list.configure(
                        values=[f'Wedge {x+1}' for x in range(len(self.em_file.wedge_size))])
                else:
                    self.process_frame.wedge_list.configure(values=[])
                    self.process_frame.update_wedge_button.config(
                        state=DISABLED)
                    self.process_frame.delete_wedge_button.config(
                        state=DISABLED)
                    self.process_frame.process_image_button.config(
                        state=DISABLED)
        else:
            selection_warning = Messagebox.show_warning(
                message='No wedge selected.\nPlease select a wedge', title='None Selected', alert=True)

    def delete_all_wedges(self):
        delete_warning = Messagebox.yesno(
            message="Do you want to delete all Wedges", title='Warning', alert=True)
        if delete_warning == "Yes":
            self.em_file.clear_mask()
            self.process_frame.wedge_list.set('')
            self.process_frame.wedge_list.configure(values=[])
            self.process_frame.process_image_button.config(bootstyle=PRIMARY)
            self.process_frame.update_wedge_button.config(state=DISABLED)
            self.process_frame.delete_wedge_button.config(state=DISABLED)
            self.process_frame.process_image_button.config(state=DISABLED)
            self.process_frame.delete_all_wedge_button.config(state=DISABLED)

    def edit_wedge(self):
        if self.process_frame.wedge_list.get() != "":
            wedge_index = np.int64(
                self.process_frame.wedge_list.get().split()[-1])-1
            self.em_file.wedge_size[wedge_index] = np.float64(
                self.process_frame.wedge_spinbox.get())
            self.em_file.wedge_angle[wedge_index] = np.float64(
                self.process_frame.theta_spinbox.get())
            self.em_file.k_min[wedge_index] = np.float64(
                self.process_frame.kmin_spinbox.get())
            self.process_frame.process_image_button.config(bootstyle=PRIMARY)
        else:
            selection_warning = Messagebox.show_warning(
                message='No wedge selected.\nPlease select a wedge', title='None Selected', alert=True)

    def view_wedge(self):
        mask = self.em_file.create_mask(np.float64(self.process_frame.wedge_spinbox.get(
        )), np.float64(self.process_frame.theta_spinbox.get()), np.float64(self.process_frame.kmin_spinbox.get()))
        self.show_wedges(mask)

    def test_process(self):
        self.em_file.process_image()
        self.em_file.reconstruct_image()
        self.setting_frame.progress_bar.stop()
        self.process_frame.process_image_button.config(bootstyle=SUCCESS)
        self.process_frame.save_image.config(state=ACTIVE)
        self.process_frame.enhance_button.config(state=ACTIVE)
        self.process_frame.view_processed_image_button.config(state=ACTIVE)

    def process_image(self):
        self.setting_frame.progress_bar.start()
        process_thread = threading.Thread(target=self.test_process)
        process_thread.start()

    def view_processed(self):
        self.show_output()
        self.show_output_fft()

    def save_processed(self):
        try:
            savefile = filedialog.asksaveasfilename(filetypes=[("TIFF file", ".tif"), ("JPEG file", ".jpg"), ("PNG file", ".png")],
                                                    defaultextension='.tif', confirmoverwrite=True)
            io.imsave(savefile, np.uint8(self.em_file.img_recon))
        except:
            pass

    def adjust_contrast(self):
        self.em_file.enhance_contrast(np.float64(
            self.process_frame.p1_spinbox.get()), np.float64(self.process_frame.p2_spinbox.get()))
        self.show_enhanced()
        self.process_frame.save_enhanced_button.config(state=ACTIVE)

    def save_enhanced(self):
        try:
            savefile = filedialog.asksaveasfilename(filetypes=[("TIFF file", ".tif"), ("JPEG file", ".jpg"), ("PNG file", ".png")],
                                                    defaultextension='.tif', confirmoverwrite=True)
            io.imsave(savefile, np.uint8(self.em_file.img_enhanced))
        except:
            pass


myapp = MyApp()
myapp.mainloop()
