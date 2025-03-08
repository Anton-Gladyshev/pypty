import tkinter as tk
from tkinter import filedialog, messagebox
import pickle
import os
import types
import inspect
import numpy as np
from pypty.initialize import append_exp_params

def lambda_to_string(f):
    if isinstance(f, types.LambdaType):
        string = inspect.getsourcelines(f)[0][0]
        first = string.find("lambda")
        if "]" in string:
            last = string.find("]")
        else:
            last = string.find(",")
        return string[first:last]
    else:
        return f


def load_params(path):
    with open(path, 'rb') as handle:
        params = pickle.load(handle)
    return params


def save_params(params_path, params):
    if params_path.endswith(".pkl"):
        try:
            os.remove(params_path)
        except:
            pass
    with open(params_path, 'wb') as file:
        pickle.dump(params, file)

def lambda_to_string(f):
    if isinstance(f, types.LambdaType):
        string=inspect.getsourcelines(f)[0][0]
        first=string.find("lambda")
        return string[first:]
    else:
        return f
#def append_exp_params(a,b):
 #   return a
def clear_placeholder(event, entry, text):
        if entry.get() == text:
            entry.delete(0, tk.END)
            entry.config(fg="black")  # Change text color to normal

def restore_placeholder(event, entry, text):
    if entry.get() == "":
        entry.insert(0, text)
        entry.config(fg="gray")  # Change text color to gray
class ParameterEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Parameter Editor")
        self.root.geometry("1000x700")

        self.params = {}
        self.entries = {}
        self.undo_stack = []
        self.redo_stack = []
        self.exp_params={
            "data_path": None,
            "masks": None,
            "path_json": "",
            "output_folder": "" ,
            "acc_voltage": None,
            "scan_size": None,
            "scan_step_A": None,
            
            "rez_pixel_size_mrad": None,
            "transform_axis_matrix": np.eye(2),
           "data_pad": None,
            
            "aberrations": np.zeros(8),
            "defocus": 0,
            
            "fov_nm": None,
            "PLRotation_deg": None,
            "data_is_numpy_and_flip_ky": False,
            "total_thickness": 1,
            "num_slices": 1,
            "print_flag": 3,
            "bright_threshold": 0.1,
            }

        button_frame = tk.Frame(root)
        button_frame.pack(fill="x", pady=5)
        self.load_button = tk.Button(button_frame, text="Load Preset", command=self.load_parameters)
        self.load_button.pack(side="left", padx=5)
        self.undo_button = tk.Button(button_frame, text="Undo", command=self.undo, state=tk.DISABLED)
        self.undo_button.pack(side="left", padx=5)
        self.redo_button = tk.Button(button_frame, text="Redo", command=self.redo, state=tk.DISABLED)
        self.redo_button.pack(side="left", padx=5)
        self.save_button = tk.Button(button_frame, text="Export", command=self.save_parameters, state=tk.DISABLED)
        self.save_button.pack(side="left", padx=5)
        
        self.refresh_button = tk.Button(button_frame, text="Save", command=self.update_params)
        self.refresh_button.pack(side="left", padx=5)
        
        
        
        self.append_button = tk.Button(button_frame, text="Append Experimental Parameters", command=self.append_exp_params_function)
        self.append_button.pack(side="left", padx=5)


        self.add_frame = tk.Frame(root)
        self.add_frame.pack(fill="x", padx=10, pady=5, anchor="w")
        entry_container = tk.Frame(self.add_frame)
        entry_container.pack(fill="x", anchor="w", padx=5)

        self.new_key_entry = tk.Entry(entry_container, width=15)
        self.new_key_entry.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        placeholder="New Parameter Name"
        self.new_key_entry.insert(0, placeholder)
        self.new_key_entry.config(fg="gray")
        self.new_key_entry.bind("<FocusIn>",  lambda e, entry=self.new_key_entry, text=placeholder: clear_placeholder(e, entry, text))
        self.new_key_entry.bind("<FocusOut>", lambda e, entry=self.new_key_entry, text=placeholder: restore_placeholder(e, entry, text)) 
        
        # Value input
        self.new_value_entry = tk.Entry(entry_container, width=20)
        self.new_value_entry.grid(row=0, column=3, padx=5, pady=2, sticky="w")
        placeholder="New Parameter Value"
        self.new_value_entry.insert(0, placeholder)
        self.new_value_entry.config(fg="gray")
        self.new_value_entry.bind("<FocusIn>",  lambda e, entry=self.new_value_entry, text=placeholder: clear_placeholder(e, entry, text))
        self.new_value_entry.bind("<FocusOut>", lambda e, entry=self.new_value_entry, text=placeholder: restore_placeholder(e, entry, text)) 
          
        
        # Add button
        self.add_button = tk.Button(entry_container, text="Add", command=self.add_entry)
        self.add_button.grid(row=0, column=4, padx=5, pady=2, sticky="w")

        self.new_key_entry.bind("<Return>", lambda event: self.add_entry())
        
        self.new_value_entry.bind("<Return>", lambda event: self.add_entry())

        # Scrollable frame setup
        self.canvas = tk.Canvas(root)
        self.scrollbar = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind scrolling
        self.canvas.bind("<Enter>", lambda e: self._bind_mousewheel())
        self.canvas.bind("<Leave>", lambda e: self._unbind_mousewheel())
        self.create_entries()
        
    

    
    
    def _on_mousewheel(self, event):
        """Enable smooth scrolling for both trackpads and mouse wheels across all OS."""
        if event.delta:  # Windows & macOS (event.delta is positive for up, negative for down)
            self.canvas.yview_scroll(-1 * (event.delta // abs(event.delta)), "units")
        elif event.num == 4:  # Linux (scroll up)
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # Linux (scroll down)
            self.canvas.yview_scroll(1, "units")

    def _bind_mousewheel(self):
        """Bind mouse wheel and touchpad scrolling for smooth behavior."""
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)  # Windows & macOS
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)  # Linux (scroll up)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)  # Linux (scroll down)

    def _unbind_mousewheel(self):
        """Unbind mouse wheel when leaving the canvas."""
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
    def load_parameters(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        if not file_path:
            return
        self.params = load_params(file_path)
        self.create_entries()
        self.save_button.config(state=tk.NORMAL)
        
    def append_exp_params_function(self):
        self.update_params(only_exp=False)
        self.params = append_exp_params(self.exp_params, self.params)
        self.create_entries()  
        self.save_button.config(state=tk.NORMAL)

    def create_entries(self):
        """Rebuilds the parameter list, displaying regular parameters on the left and experimental parameters on the right with headers."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.entries.clear()

        # 🛑 Add Headers
        tk.Label(self.scrollable_frame, text="PyPty Preset", font=("Arial", 15, "bold"), fg="#000").grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        tk.Label(self.scrollable_frame, text="Experimental Parameters", font=("Arial", 15, "bold"), fg="#000").grid(row=0, column=4, columnspan=2, padx=5, pady=5, sticky="w")
  
        row_idx = 1  # Start below the headers
        for key, value in self.params.items():
            label = tk.Label(self.scrollable_frame, text=key, fg="#222", cursor="hand2", width=25, anchor="w")
            label.grid(row=row_idx, column=0, padx=5, pady=2, sticky="w")
            label.bind("<Button-1>", lambda e, k=key, lbl=label: self.copy_to_clipboard(k, lbl))

            entry = tk.Entry(self.scrollable_frame, width=20)
            entry.grid(row=row_idx, column=1, padx=5, pady=2, sticky="w")
            
            if isinstance(value, np.ndarray):
                truncated_str = np.array2string(value, threshold=40)  # Truncate display
                entry.insert(0, truncated_str)  
                entry.full_value = value  # Store full array internally
            else:
                entry.insert(0, repr(value))

            
            self.entries[key] = entry

            delete_button = tk.Button(self.scrollable_frame, text="x", fg="red", width=2, command=lambda k=key: self.delete_entry(k))
            delete_button.grid(row=row_idx, column=2, padx=5, pady=2, sticky="w")

            row_idx += 1

        
        canvas = tk.Canvas(self.scrollable_frame, width=2, height=500, bg="gray")  # Vertical line
        canvas.grid(row=0, column=3, rowspan=100, sticky="ns", padx=10)  
        # 🛑 Add Regular Parameters in the Left Column
        
        # 🛑 Add Experimental Parameters in the Right Column
        row_idx = 1  # Start below the headers
        self.exp_entries = {}  # Store experimental entries separately

        for key, value in self.exp_params.items():
            exp_label = tk.Label(self.scrollable_frame, text=key, fg="#444", width=25, anchor="w")
            exp_label.grid(row=row_idx, column=4, padx=5, pady=2, sticky="w")
            exp_label.bind("<Button-1>", lambda e, k=key, lbl=exp_label: self.copy_to_clipboard(k, lbl))
            exp_entry = tk.Entry(self.scrollable_frame, width=20)
            exp_entry.grid(row=row_idx, column=5, padx=5, pady=2, sticky="w")
            repr_value=repr(value)
            if repr_value[:5]=="array": repr_value="np."+repr_value
            exp_entry.insert(0, repr_value)
            self.exp_entries[key] = exp_entry  # Store experimental parameters separately

            row_idx += 1

  
    def copy_to_clipboard(self, key, label):
        """Copy key to clipboard and temporarily change label color."""
        self.root.clipboard_clear()
        self.root.clipboard_append(key)
        self.root.update()

        # Change label color temporarily
        label.config(fg="#888")
        self.root.after(500, lambda: label.config(fg="#222"))  # Reset after 300ms

    def add_entry(self):
        """Adds a new entry and ensures it appears in the parameter list."""
        key = self.new_key_entry.get().strip()
        value = self.new_value_entry.get().strip()

        if not key or key in self.params:
            return
        
        if not("lambda" in value):
            try:
                parsed_value = eval(value, {
                    "np": np,
                    "array": np.array,
                    "ones": np.ones,
                    "zeros": np.zeros,
                    "load": np.load
                })

                # 🛑 Ensure NumPy arrays remain arrays
                if isinstance(parsed_value, np.ndarray):
                    parsed_value = np.array(parsed_value)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to add parameter: {e}")
                parsed_value = value  # Store as string if parsing fails
        else:
            parsed_value = value  
        
     
        self.undo_stack.append(("add", key, parsed_value))
        self.redo_stack.clear()

        self.params[key] = parsed_value
        self.create_entries()  # Refresh UI

        # Clear input fields
        self.new_key_entry.delete(0, tk.END)
        self.new_value_entry.delete(0, tk.END)

        self.undo_button.config(state=tk.NORMAL)

    def delete_entry(self, key):
        if key in self.params:
            deleted_value = self.params[key]
            self.undo_stack.append(("delete", key, deleted_value))
            self.redo_stack.clear()

            del self.params[key]
            self.create_entries()
            self.undo_button.config(state=tk.NORMAL)

    def undo(self):
        if not self.undo_stack:
            return

        action = self.undo_stack.pop()
        self.redo_stack.append(action)

        if action[0] == "add":
            del self.params[action[1]]
        elif action[0] == "delete":
            self.params[action[1]] = action[2]

        self.create_entries()
        self.undo_button.config(state=tk.NORMAL if self.undo_stack else tk.DISABLED)
        self.redo_button.config(state=tk.NORMAL)

    def redo(self):
        if self.redo_stack:
            self.undo()
            
    def update_params(self, only_exp=False):
        if not(only_exp):
            for key, entry in self.entries.items():
                value = entry.get().strip()

                try:
                    if hasattr(entry, "full_value"):  # If the full array exists, use it
                        parsed_value = entry.full_value
                    elif value.startswith("array(") or value.startswith("np.array("):
                        parsed_value = np.array(eval(value.replace("array", "np.array"), {"np": np}))
                    else:
                        parsed_value = eval(value, { "np": np, "array": np.array, "ones": np.ones, "zeros": np.zeros, "load": np.load })
                except Exception:
                    parsed_value = value  # If parsing fails, store as a string

                self.params[key] = parsed_value
    
        for key, entry in self.exp_entries.items():
            value = entry.get().strip()
            try:
                parsed_value = eval(value, { "np": np, "array": np.array, "ones": np.ones, "zeros": np.zeros, "load": np.load })
            except Exception:
                parsed_value = value  # If parsing fails, store as a string
            self.exp_params[key] = parsed_value    
        self.create_entries()  # Refresh UI
    def save_parameters(self):
        """Saves the parameters to a .pkl file, ensuring correct data types."""
        file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle Files", "*.pkl")])
        if not file_path:
            return

        try:
            for key, entry in self.entries.items():
                value = entry.get().strip()

                try:
                    if hasattr(entry, "full_value"):  # If the full array exists, use it
                        parsed_value = entry.full_value
                    elif value.startswith("array(") or value.startswith("np.array("):
                        parsed_value = np.array(eval(value.replace("array", "np.array"), {"np": np}))
                    else:
                        parsed_value = eval(value, { "np": np, "array": np.array, "ones": np.ones, "zeros": np.zeros, "load": np.load })
                except Exception:
                    parsed_value = value  # If parsing fails, store as a string
                    
                self.params[key] = parsed_value

            save_params(file_path, self.params)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save parameters: {e}")



if __name__ == "__main__":
    root = tk.Tk()
    app = ParameterEditor(root)
    root.mainloop()

