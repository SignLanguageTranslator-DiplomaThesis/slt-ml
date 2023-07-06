import tkinter as tk
from tkinter import ttk
from csv_parser.csv_parser import CsvParser


class UserInterface:
    def __init__(self):
        self._selected_label_index = 0

    # DF: creates an Input Text field, in which the user types the name of a new sign gesture label
    def generate_input_field(self, sign_language_labels):
        root = tk.Tk(screenName="Create a new label")
        root.title("Create a new label")

        ttk.Style().theme_use('clam')  # Set the theme

        # Create a frame to hold the widgets
        frame = ttk.Frame(root, padding=20)
        frame.pack()

        text_label = ttk.Label(frame, text="Enter the new label for a gesture:", font=('Arial', 12))
        text_label.grid(row=0, column=0, sticky='w', padx=10, pady=10)

        text_entry = ttk.Entry(frame, width=30, font=('Arial', 12))
        text_entry.grid(row=1, column=0, padx=10, pady=5)

        def save_label(labels):
            text = text_entry.get()
            CsvParser.write_label_to_csv([text], labels)
            text_entry.delete(0, tk.END)
            root.destroy()

        save_button = ttk.Button(frame, text="Save", command=lambda: save_label(sign_language_labels))
        save_button.grid(row=2, column=0, pady=10)

        root.mainloop()

    # DF: generates a dropdown containing all the existent sign gesture labels from the CSV file
    def generate_label_dropdown(self, sign_language_labels):
        root = tk.Tk(screenName="Choose a label")
        root.title("Choose a label")
        ttk.Style().theme_use('clam')  # Set the theme

        # Create a frame to hold the widgets
        frame = ttk.Frame(root, padding=20)
        frame.pack()

        # Create label for dropdown list
        label = ttk.Label(frame, text="Select a label:", font=('Arial', 12))
        label.grid(row=0, column=0, sticky='w', padx=10, pady=10)

        # Create dropdown list
        selected_label = tk.StringVar()
        dropdown = ttk.Combobox(frame, width=27, textvariable=selected_label, values=sign_language_labels,
                                state='readonly', font=('Arial', 12))
        dropdown.current(0)
        dropdown.grid(row=1, column=0, padx=10, pady=5)

        # Create save button
        def get_label():
            lbl = selected_label.get()
            self._selected_label_index = sign_language_labels.index(lbl)
            root.destroy()

        save_button = ttk.Button(frame, text="Save", command=get_label)
        save_button.grid(row=2, column=0, pady=10)

        root.mainloop()

    # getter
    @property
    def selected_label_index(self):
        return self._selected_label_index

