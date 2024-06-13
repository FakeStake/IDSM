import tkinter as tk
import ctypes
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from main import *
from utils import *
import threading

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("UnsteadyFlowSolver")
        self.entry_var = tk.StringVar()
        self.user_defined_widgets = [] 
        self.pitch_rate_entry = None
        self.nstep=0

        # Maximize the window
        self.root.state('zoomed')
        ctypes.windll.shcore.SetProcessDpiAwareness(0)
        
        # Hardcoded variable for the geometry file
        self.geometry_file_var = tk.StringVar(value="")
        self.ndiv_var = tk.StringVar()  # Variable for Number of Divisions\
        self.density_var = tk.StringVar(value = '1.224')
        self.velocity_var = tk.StringVar()
        self.pivot_point_var = tk.StringVar()
        self.LESP_critical_var = tk.StringVar()
        self.LESP_count=False
        self.pitch_rate_var = tk.StringVar()
        self.amplitude_var = tk.StringVar()
        self.delay_var = tk.StringVar()
        self.mean_var = tk.StringVar()
        self.noc_var = tk.StringVar()
        self.Phi_var = tk.StringVar()
        self.camber_var = tk.StringVar(value="Linear")  # Set default value to "Linear"
        self.motion_type_var=tk.StringVar()
        self.pitch_rate_entry=None
        self.amp_entry=None
        self.delay_entry=None
        self.mean_entry=None
        self.noc_entry=None
        self.Phi_entry=None 
        self.mtype=""
        self.dtstar_var=tk.StringVar(value='0.015')
        self.nsteps = 0

        self.geometry_loc=tk.StringVar()
        self.file_loc=tk.StringVar()

        global current_state_marker
        self.current_state_marker = None

        # Variables for force data options
        self.save_force_data_var = False
        self.save_flow_data_var = False

        self.save_force_data = tk.StringVar()
        self.file_force_data = tk.StringVar()
        self.tev_file_name_var = tk.StringVar()

        self.ani_count_var = tk.StringVar()

        # Variable to store geometry coordinates
        self.geometry_coordinates = tk.StringVar()
        self.create_widgets()
        self.create_example_graphs()  # Added to display graphs
        # Bind the closing event of the window to a method
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        # Add code to stop any background processes or threads here
        response = messagebox.askyesno("Confirmation", "Do you want to close the application?")
        # Check the user's response
        if response:
            exit()
    
    def create_tooltip(self,widget, text):
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25

            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{x}+{y}")
            label = ttk.Label(tooltip, text=text, background="#ffffff", relief="solid", borderwidth=1)
            label.pack(ipadx=5, ipady=3)

            widget.tooltip = tooltip  # Save tooltip reference to the widget

        def leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip  # Remove saved tooltip reference

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def create_widgets(self):
        self.create_menu_bar()

        self.options_frame_top = ttk.Frame(self.root, padding=(10, 10, 10, 0))
        self.options_frame_top.pack(side="top", fill="x")

        self.input_frame = ttk.Frame(self.root, padding=(10, 10, 10, 10))
        self.input_frame.pack(side="left", fill="both")

        options = ["Geometry", "Motion", "Operations"]  # Exchanged "Motion" and "Geometry"
        for option in options:
            button = ttk.Button(self.options_frame_top, text=option, command=lambda o=option: self.show_sub_options(o))
            button.pack(side="left", padx=5)

        # Add Exit button next to Operations button
        exit_button = ttk.Button(self.options_frame_top, text="Exit", command=self.on_close)
        exit_button.pack(side="left", padx=5)

        self.update_sub_options(options[0])

        # Create a frame for graphs on the right side
        self.graph_frame = ttk.Frame(self.root, padding=(10, 10, 10, 10))
        self.graph_frame.pack(side="right", fill="both", expand=True)

    def create_menu_bar(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

    def upload_geometry_file(self):
        file_path = filedialog.askopenfilename(title="Upload Geometry File", filetypes=[("DAT Files", "*.dat"), ("Text Files", "*.txt"), ("All Files", "*.*")])
        self.geometry_loc=file_path
        if file_path:
            self.geometry_file_var.set("Upload a Geometry")  # Set to default option
            self.read_geometry_coordinates(file_path)

    def read_geometry_coordinates(self, file_path):
        # Read geometry coordinates from the file and store them in self.geometry_coordinates
        try:
            with open(file_path, 'r') as file:
                # Assuming a simple format with two columns (x, y)
                self.geometry_coordinates = [tuple(map(float, line.split())) for line in file]
        except Exception as e:
            print(f"Error reading geometry coordinates: {e}")

        # Update the geometry plot with the new coordinates
        self.update_geometry_plot()

    def generate_geometry(self):
        # Call the appropriate methods to update the plots
        self.update_geometry_plot()
        self.plot_camber_line()


    def get_all_sub_options(self):
        sub_options = {
            "Geometry": ["Upload a Geometry", "Number of Divisions", "Camber Calculation"],
            "Motion": ["Motion Type","Pitch Rate", "Amplitude", "Delay","Phi","Mean", "Number of cycles", "dtstar", "Density", "Velocity", "Pivot Point", "LESP Critical"],
            "Operations": ["Save Force Data:", "Save Flow Data:"]
        }
        return sub_options

    def update_sub_options(self, option):
        # Clear existing input widgets
        self.clear_input_widgets()
        sub_options = self.get_all_sub_options().get(option, [])
        self.create_input_widgets(sub_options)

    def create_input_widgets(self, sub_options):
        input_box = ttk.Frame(self.input_frame, borderwidth=2, relief="groove")
        input_box.pack(side="left", padx=5, pady=5, fill="both", expand=True)
        self.input_entries = {}
        checkbox_states = {}

        def handle_motion_type(selected_option):
            if selected_option == "User defined":
                # Upload motion file and disable input bars
                self.upload_motion_file()
                self.mtype = "User defined"
                self.pitch_rate_entry.delete(0, "end")  # Clear content of pitch rate entry
                self.amp_entry.delete(0, "end")  # Clear content of amplitude entry
                self.delay_entry.delete(0, "end")  # Clear content of delay entry
                self.mean_entry.delete(0, "end")  # Clear content of mean entry
                self.Phi_entry.delete(0, "end")  # Clear content of Phi entry
                self.pitch_rate_entry.config(state='disabled')
                self.amp_entry.config(state="disabled")
                self.delay_entry.config(state='disabled')
                self.mean_entry.config(state='disabled')
                self.noc_entry.delete(0, "end")  # Clear content of noc entry
                self.noc_entry.config(state='disabled')
                self.Phi_entry.config(state='disabled')
            elif selected_option == "Pitch ramp motion":
                # Enable input bars and set motion type
                self.pitch_rate_entry.config(state='normal')
                self.amp_entry.config(state='normal')
                self.delay_entry.config(state='normal')
                self.Phi_entry.delete(0, "end")  # Clear content of Phi entry
                self.mean_entry.delete(0, "end")  # Clear content of mean entry
                self.noc_entry.delete(0, "end")  # Clear content of noc entry
                self.noc_entry.config(state='disabled')
                self.Phi_entry.config(state='disabled')
                self.mean_entry.config(state='disabled')
                self.mtype = "Pitch ramp motion"
            elif selected_option in ("Sine motion"):
                # Enable input bars and set motion type
                self.pitch_rate_entry.config(state='normal')
                self.amp_entry.config(state='normal')
                self.delay_entry.delete(0,"end")
                self.noc_entry.config(state='normal')
                self.delay_entry.config(state='disabled')
                self.Phi_entry.config(state='normal')
                self.mean_entry.config(state='normal')
                self.mtype = "Sine motion"
            elif selected_option in ("Cosine motion"):
                # Enable input bars and set motion type
                self.pitch_rate_entry.config(state='normal')
                self.amp_entry.config(state='normal')
                self.delay_entry.delete(0,"end")
                self.delay_entry.config(state='disabled')
                self.noc_entry.config(state='normal')
                self.Phi_entry.config(state='normal')
                self.mean_entry.config(state='normal')
                self.mtype = "Cosine motion"
            else:
                self.pitch_rate_entry.delete(0, "end")  # Clear content of pitch rate entry
                self.amp_entry.delete(0, "end")  # Clear content of amplitude entry
                self.delay_entry.delete(0, "end")  # Clear content of delay entry
                self.mean_entry.delete(0, "end")  # Clear content of mean entry
                self.Phi_entry.delete(0, "end")  # Clear content of Phi entry
                self.noc_entry.delete(0, "end")  # Clear content of noc entry
                self.noc_entry.config(state='disabled')
                self.pitch_rate_entry.config(state='disabled')
                self.amp_entry.config(state="disabled")
                self.delay_entry.config(state='disabled')
                self.mean_entry.config(state='disabled')
                self.Phi_entry.config(state='disabled')

        row = 0
        self.help_icon = tk.PhotoImage(file="help_icon.png")
        for sub_option in sub_options:
            if sub_option == "Upload a Geometry":
                # If "Upload a Geometry" option is selected, create a dropdown menu
                label = ttk.Label(input_box, text="Geometry Option:")
                label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

                geometry_options = ["Upload a Geometry", "FlatPlate"]
                geometry_option_menu = ttk.OptionMenu(input_box, self.geometry_file_var, "Select a Geometry", *geometry_options, command=self.handle_geometry_option)
                geometry_option_menu.grid(row=row, column=1, padx=5, pady=5, sticky="w")

                self.input_entries[sub_option] = geometry_option_menu

            elif sub_option == "Number of Divisions":
                # If "Number of Divisions" option is selected, create an Entry widget
                label = ttk.Label(input_box, text="Number of Divisions:")
                label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

                entry = ttk.Entry(input_box, textvariable=self.ndiv_var)
                entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")

                if not entry.get():
                    # If the Entry widget is empty, insert the default value
                    entry.delete(0, "end")
                    entry.insert(0, "101")  # Insert the default text
                
                self.help_label = ttk.Label(input_box, image=self.help_icon)
                self.help_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")
                self.create_tooltip(self.help_label, "Enter the number of division for the camber line")

                self.input_entries[sub_option] = entry

            elif sub_option == "Camber Calculation":
                # If "Camber Calculation" option is selected, create a dropdown menu
                label = ttk.Label(input_box, text="Camber Calculation:")
                label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

                camber_options = ["Linear", "Radial"]
                camber_option_menu = ttk.OptionMenu(input_box, self.camber_var, "Linear", *camber_options)
                camber_option_menu.grid(row=row, column=1, padx=5, pady=5, sticky="w")

                self.help_label = ttk.Label(input_box, image=self.help_icon)
                self.help_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")
                self.create_tooltip(self.help_label, "Select the type of equation to calculate the camber")

                self.input_entries[sub_option] = camber_option_menu

                generate_button = ttk.Button(input_box, text="Generate camber",command=self.generate_geometry)
                generate_button.grid(row=row+1, column=1, padx=5, pady=5, sticky="w")

            elif sub_option == "Motion Type":
                label = ttk.Label(input_box, text="Motion Type:")
                label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

                motion_type_options = ["Pitch ramp motion", "Sine motion", "Cosine motion", "User defined"]
                if self.motion_type_var.get() == "":
                    motion_type_var = tk.StringVar(value=self.motion_type_var.get())
                    motion_type_menu = ttk.OptionMenu(input_box, motion_type_var, "Select Motion Type", *motion_type_options, command=lambda opt=sub_option: handle_motion_type(opt))
                else:
                    motion_type_var = tk.StringVar(value=self.motion_type_var.get())
                    motion_type_menu = ttk.OptionMenu(input_box, motion_type_var, self.motion_type_var.get(), *motion_type_options, command=lambda opt=sub_option: handle_motion_type(opt))
                motion_type_menu.grid(row=row, column=1, padx=5, pady=5, sticky="w")
                self.motion_type_var=motion_type_var

                self.input_entries[sub_option] = motion_type_var
            elif sub_option == "Density":
                label = ttk.Label(input_box, text="Density:")
                label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

                density_entry = ttk.Entry(input_box, textvariable=self.density_var)
                density_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")

                label = ttk.Label(input_box, text="kg/m³")
                label.grid(row=row, column=2, padx=5, pady=5, sticky="e")

                if not density_entry.get():
                    density_entry.delete(0, "end")
                    density_entry.insert(0, "1.224")  # Insert the default text

                self.help_label = ttk.Label(input_box, image=self.help_icon)
                self.help_label.grid(row=row, column=3, padx=5, pady=5, sticky="w")
                self.create_tooltip(self.help_label, "Enter the value for density, if not the default value for density will be 1.224")

                self.input_entries[sub_option] = density_entry
            
            elif sub_option == "dtstar":
                label = ttk.Label(input_box, text="dtstar:")
                label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

                dtstar_entry = ttk.Entry(input_box, textvariable=self.dtstar_var)
                dtstar_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")

                if not dtstar_entry.get():
                    dtstar_entry.delete(0, "end")
                    dtstar_entry.insert(0, "0.015")  # Insert the default text

                self.help_label = ttk.Label(input_box, image=self.help_icon)
                self.help_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")
                self.create_tooltip(self.help_label, "Enter the value for dtstar, if not the default value for density will be 0.015")

                self.input_entries[sub_option] = dtstar_entry

            elif sub_option == "Velocity":
                label = ttk.Label(input_box, text="Velocity:")
                label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

                velocity_entry = ttk.Entry(input_box, textvariable=self.velocity_var )
                velocity_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")

                if not velocity_entry.get():
                    velocity_entry.delete(0, "end")
                    velocity_entry.insert(0, "1")  # Insert the default text

                self.help_label = ttk.Label(input_box, image=self.help_icon)
                self.help_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")
                self.create_tooltip(self.help_label, "Enter the value for velocity, if not the default value for velocity will be 1")

                self.input_entries[sub_option] = velocity_entry

            elif sub_option == "Pivot Point":
                label = ttk.Label(input_box, text="Pivot Point:")
                label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
                
                pivot_point_entry = ttk.Entry(input_box, textvariable=self.pivot_point_var )
                pivot_point_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")

                if not pivot_point_entry.get():
                    pivot_point_entry.delete(0, "end")
                    pivot_point_entry.insert(0, "0.25")  # Insert the default text

                self.help_label = ttk.Label(input_box, image=self.help_icon)
                self.help_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")
                self.create_tooltip(self.help_label, "Enter the pivot point, if not the default value for pivot point will be quater chord")

                self.input_entries[sub_option] = pivot_point_entry

            elif sub_option == "LESP Critical":
                def toggle_entry_state(checkbox_var, LESP_entry):
                    if checkbox_var.get():
                        LESP_entry.config(state="normal")
                        self.LESP_count = True
                    else:
                        LESP_entry.delete(0, "end")
                        LESP_entry.insert(0, "")
                        LESP_entry.config(state="disabled")

                checkbox_var = tk.BooleanVar(value=self.LESP_count)
                checkbox = ttk.Checkbutton(input_box, text="LEV shedding", variable=checkbox_var, command=lambda: toggle_entry_state(checkbox_var, LESP_entry))
                checkbox.grid(row=row, column=0, padx=5, pady=5, sticky="w")

                label = ttk.Label(input_box, text="LESP critical:")
                label.grid(row=row+1, column=0, padx=5, pady=5, sticky="e")

                LESP_entry = ttk.Entry(input_box, state="disabled", textvariable=self.LESP_critical_var)
                LESP_entry.grid(row=row+1, column=1, padx=5, pady=5, sticky="w")
                LESP_entry.delete(0, "end")
                LESP_entry.insert(0, "")


                self.help_label = ttk.Label(input_box, image=self.help_icon)
                self.help_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")
                self.create_tooltip(self.help_label, "Select the value for viscous flows, and enter the value for LESP")

                toggle_entry_state(checkbox_var, LESP_entry)

                self.input_entries[sub_option] = checkbox_var
                self.input_entries[f"{sub_option}_value"] = LESP_entry

                clear_button = ttk.Button(input_box, text="Clear All", command=self.clear_all)
                clear_button.grid(row=row+2, column=0, columnspan=3, padx=5, pady=5, sticky="sw")

            elif sub_option == "Pitch Rate":
                label_pitch_rate = ttk.Label(input_box, text="Pitch Rate:")
                label_pitch_rate.grid(row=row, column=0, padx=5, pady=5, sticky="e")

                pitch_rate_entry= ttk.Entry(input_box, textvariable=self.pitch_rate_var )
                pitch_rate_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
                self.pitch_rate_entry=pitch_rate_entry

                self.help_label = ttk.Label(input_box, image=self.help_icon)
                self.help_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")
                self.create_tooltip(self.help_label, "Enter the pitch rate")

                if self.motion_type_var.get() == "Pitch ramp motion" or self.motion_type_var.get() == "Sine motion" or self.motion_type_var.get() == "Cosine motion":
                    pitch_rate_entry.config(state='enabled')
                else:
                    pitch_rate_entry.config(state='disabled')

                self.input_entries[sub_option] = pitch_rate_entry

            elif sub_option == "Amplitude":
                label = ttk.Label(input_box, text="Amplitude:")
                label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
                
                amp_entry = ttk.Entry(input_box, textvariable=self.amplitude_var )
                amp_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
                self.amp_entry=amp_entry

                self.help_label = ttk.Label(input_box, image=self.help_icon)
                self.help_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")
                self.create_tooltip(self.help_label, "Enter the amplitude")

                if self.motion_type_var.get() == "Pitch ramp motion" or self.motion_type_var.get() == "Sine motion" or self.motion_type_var.get() == "Cosine motion":
                    amp_entry.config(state='enabled')
                else:
                    amp_entry.config(state='disabled')

                self.input_entries[sub_option] = amp_entry

            elif sub_option == "Delay":
                label = ttk.Label(input_box, text="Delay:")
                label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

                delay_entry = ttk.Entry(input_box, textvariable=self.delay_var )
                delay_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
                self.delay_entry=delay_entry

                self.help_label = ttk.Label(input_box, image=self.help_icon)
                self.help_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")
                self.create_tooltip(self.help_label, "Enter the delay for the run")

                if self.motion_type_var.get() == "Pitch ramp motion":
                    delay_entry.config(state='enabled')
                else:
                    delay_entry.config(state='disabled')

                self.input_entries[sub_option] = delay_entry

            elif sub_option == "Phi":
                label = ttk.Label(input_box, text="Phi:")
                label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

                Phi_entry = ttk.Entry(input_box, textvariable=self.Phi_var)
                Phi_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
                self.Phi_entry=Phi_entry

                self.help_label = ttk.Label(input_box, image=self.help_icon)
                self.help_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")
                self.create_tooltip(self.help_label, "Enter the value of phi")

                if self.motion_type_var.get() == "Sine motion" or self.motion_type_var.get() == "Cosine motion":
                    Phi_entry.config(state='enabled')
                else:
                    Phi_entry.config(state='disabled')

                self.input_entries[sub_option] = Phi_entry

            elif sub_option == "Mean":
                label = ttk.Label(input_box, text="Mean:")
                label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

                mean_entry = ttk.Entry(input_box, textvariable=self.mean_var )
                mean_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
                self.mean_entry=mean_entry

                self.help_label = ttk.Label(input_box, image=self.help_icon)
                self.help_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")
                self.create_tooltip(self.help_label, "Enter the mean value for the run")

                if self.motion_type_var.get() == "Sine motion" or self.motion_type_var.get() == "Cosine motion":
                    mean_entry.config(state='enabled')
                else:
                    mean_entry.config(state='disabled')

                self.input_entries[sub_option] = mean_entry
            
            elif sub_option == "Number of cycles":
                label = ttk.Label(input_box, text="Number of cycles:")
                label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

                noc_entry = ttk.Entry(input_box, textvariable=self.noc_var )
                noc_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
                self.noc_entry=noc_entry

                self.help_label = ttk.Label(input_box, image=self.help_icon)
                self.help_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")
                self.create_tooltip(self.help_label, "Enter the number of cycles for the motion")

                if self.motion_type_var.get() == "Sine motion" or self.motion_type_var.get() == "Cosine motion":
                    noc_entry.config(state='normal')
                else:
                    noc_entry.config(state='disabled')
                    noc_entry.delete(0, "end")

                self.input_entries[sub_option] = noc_entry

                # Add "Generate Motion" button after Phi
                generate_button = ttk.Button(input_box, text="Generate Motion", command=self.plot_motion_data)
                generate_button.grid(row=row+1, column=1, padx=5, pady=5, sticky="w")

                row += 2  # Increment row to move to the next row after Phi and the button

            elif sub_option.startswith("Save Force"):
                checkbox_var = tk.BooleanVar(value=self.save_force_data_var)
                checkbox = ttk.Checkbutton(input_box, text=sub_option, variable=checkbox_var, command=lambda opt=sub_option: toggle_filename_entry(opt))
                checkbox.grid(row=row, column=0, padx=5, pady=5, sticky="w")
                self.input_entries[sub_option] = checkbox_var

                # Create filename entry
                filename_label = ttk.Label(input_box, text="File Name:")
                filename_label.grid(row=row + 1, column=0, padx=5, pady=5, sticky="e")

                filename_entry = ttk.Entry(input_box, state="normal" if checkbox_var.get() else "disabled", textvariable=self.save_force_data)
                filename_entry.grid(row=row + 1, column=1, padx=5, pady=5, sticky="w")

                self.help_label = ttk.Label(input_box, image=self.help_icon)
                self.help_label.grid(row=row, column=1, padx=5, pady=5, sticky="w")
                self.create_tooltip(self.help_label, "Entry the name for force files for the run")

                self.input_entries[f"{sub_option}_filename"] = filename_entry
                row+=1

            elif sub_option.startswith("Save Flow"):
                checkbox_var = tk.BooleanVar(value=self.save_flow_data_var)
                checkbox = ttk.Checkbutton(input_box, text=sub_option, variable=checkbox_var, command=lambda opt=sub_option: toggle_filename_entry(opt))
                checkbox.grid(row=row, column=0, padx=5, pady=5, sticky="w")
                self.input_entries[sub_option] = checkbox_var

                # Create filename entry
                filename_label = ttk.Label(input_box, text="File Name:")
                filename_label.grid(row=row + 1, column=0, padx=5, pady=5, sticky="e")

                filename_entry = ttk.Entry(input_box, state="normal" if checkbox_var.get() else "disabled", textvariable=self.tev_file_name_var)
                filename_entry.grid(row=row + 1, column=1, padx=5, pady=5, sticky="w")

                # Data_out file name entry
                Data_out_filename_label = ttk.Label(input_box, text="Data Output Frequence:")
                Data_out_filename_label.grid(row=row + 2, column=0, padx=5, pady=5, sticky="e")

                Data_out_filename_entry = ttk.Entry(input_box, state="normal" if checkbox_var.get() else "disabled", textvariable=self.ani_count_var)
                Data_out_filename_entry.grid(row=row + 2, column=1, padx=5, pady=5, sticky="w")
                
                self.help_label = ttk.Label(input_box, image=self.help_icon)
                self.help_label.grid(row=row, column=1, padx=5, pady=5, sticky="w")
                self.create_tooltip(self.help_label, "Enter the name of flow files for the run that includes: \n\n\n >Leading edge vorticies, Trailing edge vorticies, and Bound vorticies in different files \n\n >Save the data for every 'n' steps")

                self.input_entries[f"{sub_option}_tev_filename"] = filename_entry
                self.input_entries[f"{sub_option}_Data_out_filename"] = Data_out_filename_entry

            else:
                # For other options, create an Entry widget
                label = ttk.Label(input_box, text=f"{sub_option}:")
                label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

                entry = ttk.Entry(input_box)
                entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")

                self.input_entries[sub_option] = entry

            row += 1
            def toggle_filename_entry(option):
                # Get the filename entry associated with the option
                filename_entry = self.input_entries.get(f"{option}_filename")
                tev_filename_entry = self.input_entries.get(f"{option}_tev_filename")
                Data_out_filename_entry = self.input_entries.get(f"{option}_Data_out_filename")

                if option == "Save Force Data:":
                    self.save_force_data_var = True

                if option == "Save Flow Data:":
                    self.save_flow_data_var = True

                if self.input_entries[option].get():
                    self.input_entries[option].set(True)

                # Toggle the state of the general filename entry based on the checkbox value
                if filename_entry:
                    filename_entry.config(state="normal" if self.input_entries[option].get() else "disabled")

                # Toggle the state of the specific filename entries based on the checkbox value
                if tev_filename_entry:
                    tev_filename_entry.config(state="normal" if self.input_entries[option].get() else "disabled")
                if Data_out_filename_entry:
                    Data_out_filename_entry.config(state="normal" if self.input_entries[option].get() else "disabled")

        # Add an "OK" button to save values
        if any(op.startswith("Save") for op in sub_options):
                # Add an "OK" button to save values
                ok_button = ttk.Button(input_box, text="RUN", command=self.save_values)
                ok_button.grid(row=row+2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
                
                label = ttk.Label(input_box, text="Summary:")
                label.grid(row=row + 3, column=0, padx=5, pady=5, sticky="w")

                label = ttk.Label(input_box, text=f"> Density : {self.density_var.get()}")
                label.grid(row=row + 4, column=0, padx=5, pady=5, sticky="w")

                label = ttk.Label(input_box, text=f"> Velocity : {self.velocity_var.get()}")
                label.grid(row=row + 5, column=0, padx=5, pady=5, sticky="w")

                label = ttk.Label(input_box, text=f"> Pivot point : {self.pivot_point_var.get()}")
                label.grid(row=row + 6, column=0, padx=5, pady=5, sticky="w")

                if self.motion_type_var.get() == "Pitch ramp motion":
                    label = ttk.Label(input_box, text=f"> Pitch rate : {self.pitch_rate_var.get()}")
                    label.grid(row=row + 7, column=0, padx=5, pady=5, sticky="w")

                    label = ttk.Label(input_box, text=f"> Amplitude : {self.amplitude_var.get()}")
                    label.grid(row=row + 8, column=0, padx=5, pady=5, sticky="w")

                    label = ttk.Label(input_box, text=f"> Delay : {self.delay_var.get()}")
                    label.grid(row=row + 9, column=0, padx=5, pady=5, sticky="w")

                if self.motion_type_var.get() == "Sine motion" or self.motion_type_var.get() == "Cosine motion":
                    label = ttk.Label(input_box, text=f"> Pitch rate : {self.pitch_rate_var.get()}")
                    label.grid(row=row + 10, column=0, padx=5, pady=5, sticky="w")

                    label = ttk.Label(input_box, text=f"> Amplitude : {self.amplitude_var.get()}")
                    label.grid(row=row + 11, column=0, padx=5, pady=5, sticky="w")

                    label = ttk.Label(input_box, text=f"> Phi : {self.Phi_var.get()}")
                    label.grid(row=row + 12, column=0, padx=5, pady=5, sticky="w")

                    label = ttk.Label(input_box, text=f"> Mean : {self.mean_var.get()}")
                    label.grid(row=row + 13, column=0, padx=5, pady=5, sticky="w")
                
                # Check if LESP_critical_var is empty and set the label text accordingly
                lesp_text = self.LESP_critical_var.get()
                if not lesp_text:  # Check if the variable is empty
                    lesp_text = "No LESP"

                label = ttk.Label(input_box, text=f"> LESP critical: {lesp_text}")
                label.grid(row=14, column=0, padx=5, pady=5, sticky="w")

                label = ttk.Label(input_box, text=f"> dtstar : {self.dtstar_var.get()}")
                label.grid(row=row + 15, column=0, padx=5, pady=5, sticky="w")

                label = ttk.Label(input_box, text=f"> Total number of steps : {self.nsteps}")
                label.grid(row=row + 16, column=0, padx=5, pady=5, sticky="w")

                self.help_label = ttk.Label(input_box, image=self.help_icon)
                self.help_label.grid(row=row+16, column=1, padx=5, pady=5, sticky="w")
                self.create_tooltip(self.help_label, "Number of steps will be updated after generating motion")

    def clear_input_widgets(self):
        for widget in self.input_frame.winfo_children():
            widget.destroy()

    def clear_all(self):
        for widget in root.winfo_children():
            widget.destroy()
        self.__init__(self.root)

    def save_values(self):
        try:
            dtstar=0.015
            # Check if the geometry file is empty
            if not isinstance(self.geometry_loc, str):
                if not self.geometry_loc.get():
                    messagebox.showerror("Error", "Please upload a geometry file.")
                    return
            
            # Check if self.mtype is empty
            if not self.mtype:
                messagebox.showerror("Error", "Please select a motion type.")
                return
            
            # If no variable is empty, proceed with saving values and calling the main function
            ndiv = self.ndiv_var.get()
            rho = self.density_var.get()
            U_inf = self.velocity_var.get()
            pvt = self.pivot_point_var.get()
            LESP = self.LESP_critical_var.get()
            k = self.pitch_rate_var.get()
            amp = self.amplitude_var.get()
            delay = self.delay_var.get()
            mean = self.mean_var.get()
            phi = self.Phi_var.get()
            motion = self.motion_type_var.get()

            save_force_data = self.save_force_data_var
            save_flow_data = self.save_flow_data_var

            file_force_data = self.save_force_data.get()
            tev_file_name = self.tev_file_name_var.get()

            # Ani count
            ani_count = self.ani_count_var.get()
            
            if LESP == "":
                LESP = "10000"

            lesp_display = "No LESP" if LESP == "10000" else LESP

            message = (
                f"> Number of Divisions: {ndiv}\n\n"
                f"> Density: {rho}\n\n"
                f"> Velocity: {U_inf}\n\n"
                f"> Pivot Point: {pvt}\n\n"
                f"> LESP Critical: {lesp_display}\n\n"
                f"> Type of Motion: {motion}\n\n"
            )
            
            if motion == "Pitch ramp motion":
                message += (
                    f"> Number of iterations: {self.nsteps}\n\n"
                    f"> Pitch Rate: {k}\n\n"
                    f"> Amplitude: {amp}\n\n"
                    f"> Delay: {delay}\n\n"
                )
            elif motion == "Sine motion" or motion == "Cosine motion":
                message += (
                    f"> Number of iterations: {self.nsteps}\n\n"
                    f"> Pitch Rate: {k}\n\n"
                    f"> Amplitude: {amp}\n\n"
                    f"> Delay: {delay}\n\n"
                    f"> Phi: {phi}\n\n"
                    f"> Mean: {mean}\n\n"
                )
            elif motion == "User defined":
                data = np.loadtxt(self.file_loc)
                message +=( f"> Number of iterations: {len(data[:,0])-2}\n\n")

            # Dictionary to map variable names to their corresponding labels
            error_messages = {}
            error_messages = {
                'ndiv_var': "> Number of divisions - More than 100 points",
                'density_var': "> Density - Should be within the range of 1 to 2",
                'velocity_var': "> Velocity - Should be 1",
                'pivot_point_var': "> Pivot point - Should be from 0 to 1",
                'LESP_critical_var': "> LESP critical - Should be from 0 to 1",
                'pitch_rate_var': "> Pitch rate - Should be below 1",
                'amplitude_var': "> Amplitude - Should be from 0 to 90",
                'delay_var': "> Delay - Should be from 10 to 15",
                'Phi_var': "> Phi - Should be from 0 to 1",
            }

            # Check for file_force_data when save_flow_data_var is True
            if self.save_force_data_var == True:
                message += (f"> Force Data: {file_force_data}\n\n")

            # Check if save_flow_data_var is True
            if self.save_flow_data_var == True:
                # Include specific error messages for TEV, and Ani_count
                message += (
                    f"> TEV File Name: {tev_file_name}\n\n"
                    f"> Ani Count: {ani_count}\n\n"
                )

           # Check if any variable is empty
            empty_variables = []
            for var_name, label in error_messages.items():
                var_value = getattr(self, var_name).get()
                if not var_value:
                    empty_variables.append(label)

            # Ani count
            ani_count = self.ani_count_var.get()

            # Check for valid ranges and update error_messages dictionary
            empty_variables = []
            # Check for empty variables and conditions simultaneously
            # Check common variables
            if ndiv == '':
                empty_variables.append("> Number of divisions is empty.")
            elif float(ndiv) > 200:
                empty_variables.append("> Number of divisions - Should be less than or equal to 200.")

            if rho == '':
                empty_variables.append("> Density is empty.")

            if U_inf == '':
                empty_variables.append("> Velocity is empty.")

            if pvt == '':
                empty_variables.append("> Pivot point is empty.")

            if LESP == '':
                empty_variables.append("> LESP critical is empty.")

            # Check motion-specific variables
            if motion == "Pitch ramp motion" or motion == "Sine motion" or motion == "Cosine motion":
                if k == '':
                    empty_variables.append("> Pitch rate is empty.")
                elif float(k) > 1:
                    empty_variables.append("> Pitch rate - Should be a number below 1.")

                if amp == '':
                    empty_variables.append("> Amplitude is empty.")
                elif not amp.isdigit() or not (0 <= float(amp) <= 360):
                    empty_variables.append("> Amplitude - Should be a number from 0 to 90.")

            elif motion == "Sine motion" or motion == "Cosine motion":
                if phi == '':
                    empty_variables.append("> Phi is empty.")
                elif not phi.isdigit() or not (0 <= float(phi) <= 1):
                    empty_variables.append("> Phi - Should be a number from 0 to 1.")

                if mean == '':
                    empty_variables.append("> Mean is empty.")
                elif not mean.isdigit():
                    empty_variables.append("> Mean - Should be a number.")

            if self.save_force_data_var == True:
                if not file_force_data:  # Check if the filename is empty
                    empty_variables.append("> Force File Name - Enter a file name.")

            if self.save_flow_data_var == True:
                # Include specific error messages for TEV, and Ani_count
                if not tev_file_name:
                    empty_variables.append("> TEV File Name - Enter a file name.")
                if not ani_count:
                    empty_variables.append("> Data Output Frequency - Enter the number of steps to save the data.")
                    

            # If any variable is empty, raise an exception in a dialog window
            if empty_variables:
                error_message = "The following fields have errors:\n\n" + "\n\n".join(empty_variables)
                messagebox.showerror("Error", error_message)
                return

            # Show a messagebox with the values and ask for confirmation
            confirmation = messagebox.askyesno("Confirmation", f"Do you want to continue with the run?\n\n{message}")

            # Close the window and go back to the main application if "No" is selected
            if confirmation:
                self.update_geometry_plot()
                self.plot_camber_line()
                self.plot_motion_data()

                if save_flow_data == True:
                    ttime=main(self.geometry_loc, self.file_loc, app, motion, ndiv, self.nsteps, rho, pvt, LESP, k, amp, delay, phi, mean, save_force_data, file_force_data, save_flow_data, tev_file_name, ani_count)
                    new_con=messagebox.showinfo("Conformation",f"The simulation is complete. The total run time is {math.floor(ttime)} seconds.")
                else:
                    ttime=main(self.geometry_loc, self.file_loc, app, motion, ndiv, self.nsteps, rho, pvt, LESP, k, amp, delay, phi, mean, save_force_data, file_force_data, save_flow_data)
                    new_con=messagebox.showinfo("Conformation",f"The simulation is complete. The total run time is {math.floor(ttime)} seconds.")
            else:
                # User clicked "No", the messagebox will simply disappear without performing any action
                print("User clicked No")


        except AttributeError as e:
            messagebox.showerror("Error", f"The following field is empty: {str(e)}")

    def show_sub_options(self, option):
        self.update_sub_options(option)

    def update_graphs(self, mat, camX, camZ, tev, lev, alpha, t):
        # Clear existing plots
      for ax in self.graphs.values():
        if ax not in [self.graphs["Geometry"], self.graphs["Motion Definition"]]:
            ax.clear()

        # Update plots
        self.graphs["LESP vs t*"].plot(mat[0][:], mat[2][:], color="black")
        self.graphs["LESP vs t*"].set_title("LESP vs t*")
        self.graphs["LESP vs t*"].set_xlabel("t*")
        self.graphs["LESP vs t*"].set_ylabel("LESP")

        self.graphs["Cl vs t*"].plot(mat[0][:], mat[3][:], color="black")
        self.graphs["Cl vs t*"].set_title("Cl vs t*")
        self.graphs["Cl vs t*"].set_xlabel("t*")
        self.graphs["Cl vs t*"].set_ylabel("Cl")

        self.graphs["Cd vs t*"].plot(mat[0][:], mat[4][:], color="black")
        self.graphs["Cd vs t*"].set_title("Cd vs t*")
        self.graphs["Cd vs t*"].set_xlabel("t*")
        self.graphs["Cd vs t*"].set_ylabel("Cd")

        self.graphs["Cm vs t*"].plot(mat[0][:], mat[5][:], color="black")
        self.graphs["Cm vs t*"].set_title("Cm vs t*")
        self.graphs["Cm vs t*"].set_xlabel("t*")
        self.graphs["Cm vs t*"].set_ylabel("Cm")
        
        global current_state_marker

        # Remove the previous state marker if it exists
        if self.current_state_marker is not None:
            self.current_state_marker.remove()
            self.current_state_marker = None

        # Plot the new state marker and keep a reference to it
        self.current_state_marker = self.graphs["Motion Definition"].plot(t,np.degrees(alpha), 'ro')[0]

        # For the animation vortex graph
        ax_bottom = self.graphs["Vortex Animation"]
        ax_bottom.set_title("Vortex animation")
        ax_bottom.set_xticks([])
        ax_bottom.set_yticks([])

        # Apply tight layout to avoid label overlapping for the empty graphs
        plt.tight_layout()

        # Set the new center coordinates
        new_center_x = camX[0]
        new_center_y = camZ[0]

        # Set the axis limits to create a new center
        ax_bottom.set_xlim(new_center_x - 8, new_center_x + 20)
        ax_bottom.set_ylim(new_center_y - 4, new_center_y + 4)

        ax_bottom.set_aspect('equal')
        
        ax_bottom.plot(camX, camZ, color="black", linewidth=2)  # Add vortex animation plot

        if (tev[0]) != [0]:
            ax_bottom.scatter(tev[0][:], tev[1][:], s=2, color='red', marker='o')

        if (lev[0]) != [0]:
            ax_bottom.scatter(lev[0][:], lev[1][:], color='green', marker='o', s=2)

        # Pack the new canvases into the Tkinter frame
        self.canvas_top.draw_idle()
        self.canvas_bottom.draw_idle()

    def create_example_graphs(self):
            # Create and display example graphs with the specified titles
            fig = plt.figure(figsize=(8, 6))

            # Create empty 6 graphs in a 2x3 grid with specified titles
            ax1 = fig.add_subplot(231)
            ax1.set_title('Geometry')
            ax1.set_xlabel('x/c')
            ax1.set_ylabel('y/c')

            ax2 = fig.add_subplot(232)
            ax2.set_title('LESP vs t*')
            ax2.set_xlabel('t*')
            ax2.set_ylabel('LESP')

            ax3 = fig.add_subplot(233)
            ax3.set_title('C\u2097 vs t*')
            ax3.set_xlabel('t*')
            ax3.set_ylabel('C\u2097')

            ax4 = fig.add_subplot(234)
            ax4.set_title('Motion Definition')
            ax4.set_xlabel('t*')
            ax4.set_ylabel(r'$\alpha$')

            ax5 = fig.add_subplot(235)
            ax5.set_title('Cd vs t*')
            ax5.set_xlabel('t*')
            ax5.set_ylabel('Cd')

            ax6 = fig.add_subplot(236)
            ax6.set_title('Cm vs t*')
            ax6.set_xlabel('t*')
            ax6.set_ylabel('Cm')

            # Apply tight layout to avoid label overlapping for the empty graphs
            plt.tight_layout()

            # Create empty 7th graph that spans the entire length of x-axis
            fig_bottom, ax_bottom = plt.subplots(figsize=(12, 4))
            ax_bottom.set_title('Vortex Animation')
            ax_bottom.set_xticks([])
            ax_bottom.set_yticks([])

            ax_bottom.set_ylim(-4, 4)
            ax_bottom.set_xlim(-8, 20)

            ax_bottom.set_aspect('equal')

            # Embed both figures into the Tkinter GUI
            self.graphs = {
                'Geometry': ax1,
                'LESP vs t*': ax2,
                'Cl vs t*': ax3,
                'Motion Definition': ax4,
                'Cd vs t*': ax5,
                'Cm vs t*': ax6,
                'Vortex Animation': ax_bottom
            }

            self.canvas_top = FigureCanvasTkAgg(fig, master=self.graph_frame)
            self.canvas_top.draw()
            self.canvas_top.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.canvas_bottom = FigureCanvasTkAgg(fig_bottom, master=self.graph_frame)
            self.canvas_bottom.draw()
            self.canvas_bottom.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def update_geometry_plot(self):
        # Update the geometry plot with the new geometry coordinates
        ax = self.graphs["Geometry"]

        if self.geometry_file_var.get() == "FlatPlate":
            ax.clear()  # Clear existing plot
            ax.plot([0, 1], [0, 0], color='black', linewidth=2, label='FlatPlate')  # Draw FlatPlate line
            ax.set_title('Geometry - FlatPlate')
            ax.set_xlabel('x/c')
            ax.set_ylabel('y/c')
            ax.set_ylim(-0.5, 0.5)  # Set y-limit for the geometry plot
            self.geometry_loc="FlatPlate"
        else:
            x_geometry, y_geometry = zip(*self.geometry_coordinates)
            ax.clear()  # Clear existing plot
            ax.plot(x_geometry, y_geometry, label='Geometry')  # Plot geometry coordinates

            # Set the title of the geometry plot as the name of the input file
            ax.set_title('Geometry - Airfoil')  # Keep the option name
            ax.set_xlabel('x/c')
            ax.set_ylabel('y/c')
            ax.legend()

            # Set y-limit for the geometry plot
            ax.set_ylim(-0.5, 0.5)

        # Apply tight layout to avoid label overlapping
        plt.tight_layout()

        # Redraw canvas to reflect changes
        self.canvas_top.draw()

    def handle_geometry_option(self, selected_option):
        if selected_option == "Upload a Geometry":
            self.upload_geometry_file()
        elif selected_option == "FlatPlate":
            self.update_geometry_plot()

    def plot_camber_line(self, *args):

        ax = self.graphs["Geometry"]

        if self.geometry_file_var.get() == "FlatPlate":
            # Clear existing camber line
            for line in ax.lines:
                if line.get_label() == 'Camber':
                    line.remove()

            # Plot the camber line for a flat plate (horizontal line)
            ax.plot([0, 1], [0, 0], color='red', linestyle='--', label='Camber')

            # Divide the camber plot into the number of divisions
            num_divisions = int(self.ndiv_var.get())
            if num_divisions > 1:
                division_points = np.linspace(0, 1, num_divisions)
                for point in division_points:
                    ax.scatter(point, 0, color='blue', marker='+', s=20)
            ax.legend()
        
        else:
            try:
                x_geometry, y_geometry = zip(*self.geometry_coordinates)
            except Exception as e: messagebox.showerror("Error", "An error occurred while processing the geometry file. Please upload a correct geometry file.")

            # Clear existing camber line
            for line in ax.lines:
                if line.get_label() == 'Camber':
                    line.remove()

            if self.camber_var.get() == "Linear":
                # Linear camber calculation (simple average)
                #camber_line_y = [(max(y_geometry[i], y_geometry[-(i+1)]) + min(y_geometry[i], y_geometry[-(i+1)])) / 2 for i in range(len(y_geometry))]
                ndiv = int(self.ndiv_var.get())
                x_geo=np.zeros(ndiv)

                x_geo = np.linspace(min(x_geometry), max(x_geometry), ndiv)
                camber_line_y,slope=camber_calc(x_geo,self.geometry_loc)
                ax.plot(x_geo, camber_line_y, label='Camber', linestyle='--', color='red')
                ax.legend()

                if ndiv > 1:
                    division_points = np.linspace(min(x_geometry), max(x_geometry), ndiv)
                    for point in division_points:
                        ax.scatter(point, np.interp(point, x_geo, camber_line_y), color='blue', marker='+', s=20,)
            else:
                # Radial camber calculation
                ndiv = int(self.ndiv_var.get())
                upper_surface = []
                lower_surface = []
                for xi, yi in zip(x_geometry, y_geometry):
                    if yi >= 0:
                        upper_surface.append((xi, yi))
                    else:
                        lower_surface.append((xi, yi))
                
                upper_surface = np.array(upper_surface)
                lower_surface = np.array(lower_surface)
                
                dtheta = np.pi / (ndiv - 1)
                theta = np.zeros(ndiv)
                x_camber = np.zeros(ndiv)
                y_camber = np.zeros(ndiv)

                for ib in range(ndiv):
                    theta[ib] = ib * dtheta
                    x_camber[ib] = max(x_geometry) / 2. * (1 - np.cos(theta[ib]))
                    y_upper_interp = np.interp(x_camber[ib], upper_surface[:, 0], upper_surface[:, 1])
                    y_lower_interp = np.interp(x_camber[ib], lower_surface[:, 0], lower_surface[:, 1])
                    y_camber[ib] = (y_upper_interp + y_lower_interp) / 2
                
                ax.plot(x_camber,-(y_camber) , label='Camber', linestyle='--', color='red')
                ax.legend()

                if ndiv > 1:
                    division_points = np.linspace(min(x_geometry), max(x_geometry), ndiv)
                    for point in division_points:
                        ax.scatter(point, np.interp(point, x_camber,-(y_camber) ), color='blue', marker='+', s=20,)

        # Apply tight layout to avoid label overlapping
        plt.tight_layout()

        # Redraw canvas to reflect changes
        self.canvas_top.draw()

    def upload_motion_file(self):
        file_path = filedialog.askopenfilename(title="Upload Motion File", filetypes=[("DAT Files", "*.dat"), ("Text Files", "*.txt"), ("All Files", "*.*")])
        self.file_loc=file_path

    def plot_motion_data(self):
        empty_vars = []  # List to store names of empty variables
        if not self.mtype == "User defined":
            # Check if amplitude_var, pitch_rate_var, delay_var, and Phi_var are empty
            if not self.amplitude_var.get():
                empty_vars.append("> Amplitude")
            if not self.pitch_rate_var.get():
                empty_vars.append("> Pitch Rate")
            if (self.mtype == "Sine motion" or self.mtype == "Cosine motion"):
                if not self.Phi_var.get():
                    empty_vars.append("> Phi")
                
                if not self.mean_var.get():
                    empty_vars.append("> Mean")

                if not self.noc_var.get():
                    empty_vars.append("> Number of cycles")

        # If any variable is empty, display an error dialog
        if empty_vars:
            error_message = "The following variables are empty:\n\n" + "\n\n".join(empty_vars)
            messagebox.showerror("Error", error_message)
            return  # Exit the function if any variable is empty
        
        if self.mtype == "Pitch ramp motion":

            dtstar=0.015
            amp=float(self.amplitude_var.get())
            k=float(self.pitch_rate_var.get())
            tstart=float(self.delay_var.get())

            fr = k/ (pi*abs(math.radians(amp)))
            t1 = tstart
            t2 = t1 + (1 /(2*pi*fr))
            t3 = t2 + ((1/(4*fr)) - (1/(2*pi*fr)))
            t4 = t3 + (1 /(2*pi*fr))
            t5 = t4 + t1
            t_tot=t5
            t = np.arange(0, t_tot, dtstar)

            self.nsteps = int(round(t_tot/dtstar))
            alpha = np.empty(self.nsteps)

            a = (np.pi**2 * k * 180) / (2 * amp * np.pi * (1 - 0.1))

            alphadef = EldRampReturntstartDef(amp* np.pi / 180, k, a, tstart, dtstar)

            for i in range (self.nsteps):
                tt=t[i]
                alpha[i]=alphadef(tt)

            ax = self.graphs["Motion Definition"]
            ax.clear()
            
            t=t[:len(alpha)]
            ax.plot(t, np.degrees(alpha) )
            ax.set_title("Motion Definition")
            ax.set_xlabel("t*")
            ax.set_ylabel(r'$\alpha$')
            
            # Redraw canvas to reflect changes
            self.canvas_top.draw()

        elif self.mtype == "Sine motion":
            noc=float(self.noc_var.get())
            mean=float(self.mean_var.get())
            dtstar=0.015

            amp=float(self.amplitude_var.get())
            k=float(self.pitch_rate_var.get())
            phi=float(self.Phi_var.get())

            rt=pi/k
            t_tot=rt*noc
            t = np.arange(0, t_tot, dtstar)
            self.nsteps = int(round(t_tot/dtstar))
            alpha=np.empty(self.nsteps)

            alphadef = SinDef(np.radians(mean), amp * np.pi / 180, k, np.radians(phi))

            for i in range (self.nsteps):
                tt=t[i]
                alpha[i]=alphadef(tt)

            ax = self.graphs["Motion Definition"]
            ax.clear()
            
            t=t[:len(alpha)]
            ax.plot(t,  np.degrees(alpha) )
            ax.set_title("Motion Definition")
            ax.set_xlabel("t*")
            ax.set_ylabel(r'$\alpha$')
            
            # Redraw canvas to reflect changes
            self.canvas_top.draw()
        elif self.mtype == "Cosine motion":
            noc=float(self.noc_var.get())
            mean=float(self.mean_var.get())
            dtstar=0.015

            amp=float(self.amplitude_var.get())
            k=float(self.pitch_rate_var.get())
            phi=float(self.Phi_var.get())

            rt=pi/k
            t_tot=rt*noc
            t = np.arange(0, t_tot, dtstar)
            self.nsteps = int(round(t_tot/dtstar))
            alpha=np.empty(self.nsteps)

            alphadef = CosDef(np.radians(mean), amp * np.pi / 180, k, np.radians(phi))

            for i in range (self.nsteps):
                tt=t[i]
                alpha[i]=alphadef(tt)

            ax = self.graphs["Motion Definition"]
            ax.clear()
            
            t=t[:len(alpha)]
            ax.plot(t,  np.degrees(alpha) )
            ax.set_title("Motion Definition")
            ax.set_xlabel("t*")
            ax.set_ylabel(r'$\alpha$')
            
            # Redraw canvas to reflect changes
            self.canvas_top.draw()

        else:
            # Clear existing motion definition plot
            ax = self.graphs["Motion Definition"]
            ax.clear()

            # Read data from .dat or .txt file
            try:
                data = np.loadtxt(self.file_loc)

                # Plot the data
                row1 = data[:,0]
                row2 = data[:,1]
                self.nsteps = len(row1)-2
                ax.plot(row1, row2 )
                ax.set_title("Motion Definition")
                ax.set_xlabel("t*")
                ax.set_ylabel(r'$\alpha$')

            except Exception as e:
                print(f"Error reading motion data from file: {e}")

            # Redraw canvas to reflect changes
            self.canvas_top.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
