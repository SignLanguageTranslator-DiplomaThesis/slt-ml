import tkinter as tk
from tkinter import ttk
from csv_parser.csv_parser import CsvParser


class UserInterface:
    def __init__(self):
        self._selected_label_index = 0

    # DF: creates an Input Text field, in which the user types the name of a new sign gesture label
    def generate_input_field(self, sign_language_labels):
        root = tk.Tk()

        text_label = tk.Label(root, text="Enter text:")
        text_label.pack()

        text_entry = tk.Entry(root)
        text_entry.pack()

        def save_label(labels):
            text = text_entry.get()
            CsvParser.write_label_to_csv([text], labels)
            text_entry.delete(0, tk.END)
            root.destroy()

        save_button = tk.Button(root, text="Save", command=lambda: save_label(sign_language_labels))
        save_button.pack()

        root.mainloop()

    # DF: generates a dropdown containing all the existent sign gesture labels from the CSV file
    def generate_label_dropdown(self, sign_language_labels):
        root = tk.Tk()

        # Create label for dropdown list
        label = tk.Label(root, text="Select a label:")
        label.pack(pady=5)

        # Create dropdown list
        selected_label = tk.StringVar()
        dropdown = ttk.Combobox(root, width=27, textvariable=selected_label, values=sign_language_labels,
                                state='readonly')
        dropdown.current(0)
        dropdown.pack(pady=5)

        # Create save button
        def get_label():
            # global number
            lbl = selected_label.get()
            self._selected_label_index = sign_language_labels.index(lbl)
            root.destroy()

        save_button = tk.Button(root, text="Save", command=get_label)
        save_button.pack(pady=5)

        # Start event loop
        root.mainloop()

    # getter
    @property
    def selected_label_index(self):
        return self._selected_label_index

