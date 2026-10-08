import os
import sys
import subprocess
import threading
import platform
import shutil
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

# Configuration
GALLERY_DIR = "./gallery"
THUMBNAIL_SIZE = (160, 160)
GRID_COLUMNS = 4

class VisualSearchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Visual Search Engine")
        self.geometry("900x700")
        self.minsize(700, 500)
        self.configure(bg="#f5f6f8")

        # Keep references to PhotoImage objects so they aren't garbage-collected
        self.image_references = []
        self.search_image_path = None

        self._setup_ui()
        self._start_ingestion_thread()

    def _setup_ui(self):
        # 1. Header Bar
        header_frame = tk.Frame(self, bg="#1e293b", height=70)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="Smart Visual Search — Image Gallery",
            font=("Segoe UI", 16, "bold"),
            fg="#ffffff",
            bg="#1e293b"
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=15)

        # 2. Status Banner
        self.status_var = tk.StringVar(value="Starting up...")
        self.status_bar = tk.Label(
            self,
            textvariable=self.status_var,
            font=("Segoe UI", 10),
            bg="#e2e8f0",
            fg="#334155",
            anchor="w",
            padx=15,
            pady=8
        )
        self.status_bar.pack(fill=tk.X, side=tk.TOP)

        # Search controls accept either or both query modalities.
        search_frame = tk.Frame(self, bg="#f5f6f8")
        search_frame.pack(fill=tk.X, padx=20, pady=(12, 0))

        self.query_var = tk.StringVar()
        self.query_entry = tk.Entry(
            search_frame,
            textvariable=self.query_var,
            font=("Segoe UI", 11),
            relief="solid",
            bd=1
        )
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=7)
        self.query_entry.insert(0, "Describe the images you want to find...")
        self.query_entry.bind("<FocusIn>", self._clear_query_placeholder)
        self.query_entry.bind("<Return>", lambda event: self._search())

        self.choose_image_button = tk.Button(
            search_frame,
            text="Choose image...",
            command=self._choose_search_image,
            font=("Segoe UI", 10),
            bg="#ffffff",
            fg="#1e293b",
            padx=10,
            pady=6,
            relief="solid",
            cursor="hand2"
        )
        self.choose_image_button.pack(side=tk.LEFT, padx=(8, 0))

        self.search_button = tk.Button(
            search_frame,
            text="Search",
            command=self._search,
            font=("Segoe UI", 10, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            padx=18,
            pady=6,
            relief="flat",
            cursor="hand2",
            state=tk.DISABLED
        )
        self.search_button.pack(side=tk.LEFT, padx=(8, 0))

        self.selected_image_label = tk.Label(
            self,
            text="No reference image selected",
            font=("Segoe UI", 9),
            bg="#f5f6f8",
            fg="#64748b",
            anchor="w"
        )
        self.selected_image_label.pack(fill=tk.X, padx=22, pady=(4, 0))

        # 3. Scrollable Canvas Area for Gallery
        container = tk.Frame(self, bg="#f5f6f8")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        self.canvas = tk.Canvas(container, bg="#ffffff", highlightthickness=1, highlightbackground="#cbd5e1")
        self.scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.gallery_frame = tk.Frame(self.canvas, bg="#ffffff")
        self.gallery_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.gallery_frame, anchor="nw")
        self.canvas.configure(xscrollcommand=None, yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Enable mouse wheel scrolling
        self.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _start_ingestion_thread(self):
        """Runs the ingestion pipeline in the background to prevent UI freeze."""
        self.status_var.set("⏳ Running ingestion pipeline (CLIP encoding & ChromaDB indexing)... please wait.")
        threading.Thread(target=self._run_ingestion, daemon=True).start()

    def _run_ingestion(self):
        try:
            if os.path.exists("ingestion_pipeline.py"):
                # Execute the ingestion pipeline script
                result = subprocess.run(
                    [sys.executable, "ingestion_pipeline.py"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(result.stdout)
            else:
                print("Notice: 'ingestion_pipeline.py' not found in current directory. Skipping execution.")
        except subprocess.CalledProcessError as e:
            print(f"Error during ingestion pipeline: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        # Once ingestion completes, load images on the main GUI thread
        self.after(0, self._on_ingestion_complete)

    def _on_ingestion_complete(self):
        self.search_button.configure(state=tk.NORMAL)
        self._load_gallery()

    def _clear_query_placeholder(self, _event):
        if self.query_var.get() == "Describe the images you want to find...":
            self.query_var.set("")

    def _choose_search_image(self):
        filepath = filedialog.askopenfilename(
            title="Choose a reference image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if filepath:
            self.search_image_path = filepath
            self.selected_image_label.configure(text=f"Reference image: {os.path.basename(filepath)}")

    def _search(self):
        text_query = self.query_var.get().strip()
        if text_query == "Describe the images you want to find...":
            text_query = ""
        if not text_query and not self.search_image_path:
            messagebox.showinfo("Search", "Enter a description, choose an image, or provide both.", parent=self)
            return

        self.search_button.configure(state=tk.DISABLED)
        self.status_var.set("Searching indexed images...")
        threading.Thread(
            target=self._run_agent_search,
            args=(text_query or None, self.search_image_path),
            daemon=True
        ).start()

    def _run_agent_search(self, text_query, image_path):
        try:
            # Import lazily and run away from Tk's UI thread; agent initialization loads CLIP.
            from agent import run_agent

            results = run_agent(text_query=text_query, image_path=image_path, k=20)
            self.after(0, lambda: self._show_search_results(results))
        except Exception as error:
            self.after(0, lambda error=error: self._search_failed(error))

    def _search_failed(self, error):
        self.search_button.configure(state=tk.NORMAL)
        self.status_var.set("Search failed.")
        messagebox.showerror("Search error", str(error), parent=self)

    def _show_search_results(self, results):
        self.search_button.configure(state=tk.NORMAL)
        if isinstance(results, str):
            self._render_gallery([], results, results)
            return

        image_items = []
        for result in results or []:
            filename = result.get("filename")
            if not filename:
                continue
            indexed_path = result.get("filepath")
            filepath = indexed_path if indexed_path and os.path.isfile(indexed_path) else os.path.join(
                GALLERY_DIR, os.path.basename(filename)
            )
            if os.path.isfile(filepath):
                image_items.append((filepath, os.path.basename(filename)))

        count = len(image_items)
        status = f"Found {count} matching image(s)." if count else "No matching images found."
        self._render_gallery(image_items, status, "No matching images found for this search.")

    def _load_gallery(self):
        """Scans the gallery directory and populates the UI grid."""
        # Ensure gallery folder exists
        if not os.path.exists(GALLERY_DIR):
            os.makedirs(GALLERY_DIR, exist_ok=True)
            self._render_gallery([], f"Created '{GALLERY_DIR}' folder. Please place .jpg/.png images there.",
                                 "No images to display. Add images to the gallery folder.")
            return

        valid_exts = (".jpg", ".jpeg", ".png")
        image_files = [f for f in sorted(os.listdir(GALLERY_DIR)) if f.lower().endswith(valid_exts)]
        if not image_files:
            self._render_gallery([], f"No images found in '{GALLERY_DIR}'. Add some images.",
                                 "No images to display.\nDrop images into the './gallery' folder and reload.")
            return

        image_items = [(os.path.join(GALLERY_DIR, filename), filename) for filename in image_files]
        self._render_gallery(image_items, f"Loaded {len(image_files)} image(s) from '{GALLERY_DIR}'. Ready.", "")

    def _render_gallery(self, image_items, status_text, empty_message):
        """Displays gallery or search-result images in the scrollable grid."""
        for widget in self.gallery_frame.winfo_children():
            widget.destroy()
        self.image_references.clear()
        self.status_var.set(status_text)

        if not image_items:
            tk.Label(
                self.gallery_frame,
                text=empty_message,
                font=("Segoe UI", 12),
                fg="#64748b",
                bg="#ffffff",
                pady=40
            ).pack()
            return

        # Populate Grid
        for idx, (filepath, filename) in enumerate(image_items):
            row = idx // GRID_COLUMNS
            col = idx % GRID_COLUMNS

            card = tk.Frame(self.gallery_frame, bg="#ffffff", bd=1, relief="solid", padx=5, pady=5)
            card.grid(row=row, column=col, padx=12, pady=12, sticky="nsew")

            try:
                # Load and thumbnail
                pil_img = Image.open(filepath)
                pil_img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(pil_img)
                self.image_references.append(tk_img)

                img_label = tk.Label(card, image=tk_img, bg="#ffffff", cursor="hand2")
                img_label.pack()

                # Filename label
                display_name = filename if len(filename) <= 18 else filename[:15] + "..."
                text_label = tk.Label(
                    card,
                    text=display_name,
                    font=("Segoe UI", 9),
                    bg="#ffffff",
                    fg="#1e293b",
                    wraplength=140,
                    cursor="hand2"
                )
                text_label.pack(pady=(4, 0))

                # Bind click events to open the preview window
                img_label.bind("<Button-1>", lambda e, fp=filepath, fn=filename: self._show_image_preview(fp, fn))
                text_label.bind("<Button-1>", lambda e, fp=filepath, fn=filename: self._show_image_preview(fp, fn))

            except Exception as e:
                err_label = tk.Label(card, text=f"Error loading:\n{filename}", font=("Segoe UI", 8), fg="red")
                err_label.pack()
                print(f"Failed to load {filename}: {e}")

    def _show_image_preview(self, filepath, filename):
        """Opens a new window displaying the full-size image with Download and Copy options."""
        top = tk.Toplevel(self)
        top.title(f"Preview - {filename}")
        top.configure(bg="#f5f6f8")
        top.geometry("800x650")

        # 1. Load and Scale Image
        try:
            img = Image.open(filepath)
            
            # Scale down if it exceeds reasonable screen dimensions
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            max_w = int(screen_w * 0.7)
            max_h = int(screen_h * 0.7)
            
            img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            
            img_lbl = tk.Label(top, image=tk_img, bg="#f5f6f8")
            img_lbl.image = tk_img  # Prevent garbage collection
            img_lbl.pack(expand=True, padx=20, pady=20)
            
        except Exception as e:
            tk.Label(top, text=f"Could not load image: {e}").pack(pady=20)
            return

        # 2. Button Controls
        btn_frame = tk.Frame(top, bg="#f5f6f8")
        btn_frame.pack(side=tk.BOTTOM, pady=20)

        dl_btn = tk.Button(
            btn_frame, text="Download Image", 
            command=lambda: self._download_image(filepath, filename, top),
            font=("Segoe UI", 10), bg="#0ea5e9", fg="white", 
            padx=15, pady=6, relief="flat", cursor="hand2"
        )
        dl_btn.pack(side=tk.LEFT, padx=10)

        cp_btn = tk.Button(
            btn_frame, text="Copy to Clipboard", 
            command=lambda: self._copy_to_clipboard(filepath, top),
            font=("Segoe UI", 10), bg="#10b981", fg="white", 
            padx=15, pady=6, relief="flat", cursor="hand2"
        )
        cp_btn.pack(side=tk.LEFT, padx=10)

    def _download_image(self, filepath, filename, parent_window):
        """Copies the image from the gallery to a user-selected destination."""
        ext = os.path.splitext(filename)[1].lower()
        filetypes = [("JPEG", "*.jpg;*.jpeg"), ("PNG", "*.png"), ("All Files", "*.*")]
        
        dst = filedialog.asksaveasfilename(
            initialfile=filename,
            defaultextension=ext,
            filetypes=filetypes,
            title="Save Image As",
            parent=parent_window
        )
        if dst:
            try:
                shutil.copy2(filepath, dst)
                messagebox.showinfo("Success", f"Image saved successfully to:\n{dst}", parent=parent_window)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{e}", parent=parent_window)

    def _copy_to_clipboard(self, filepath, parent_window):
        """Cross-platform method to copy an image file to the system clipboard."""
        system = platform.system()
        abs_path = os.path.abspath(filepath)
        success = False

        try:
            if system == "Windows":
                # Escape single quotes in path for PowerShell
                safe_path = abs_path.replace("'", "''")
                
                # Fetch cross-version compatible no-window flag
                creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
                cmd = [
                    "powershell", "-command",
                    f"Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Clipboard]::SetImage([System.Drawing.Image]::FromFile('{safe_path}'))"
                ]
                subprocess.run(cmd, check=True, creationflags=creationflags)
                success = True

            elif system == "Darwin":  # macOS
                safe_path = abs_path.replace('"', '\\"')
                # Read as TIFF picture for maximum compatibility across macOS image APIs
                cmd = ["osascript", "-e", f'set the clipboard to (read (POSIX file "{safe_path}") as TIFF picture)']
                subprocess.run(cmd, check=True)
                success = True

            elif system == "Linux":
                import mimetypes
                mime, _ = mimetypes.guess_type(abs_path)
                mime = mime or "image/png"
                cmd = ["xclip", "-selection", "clipboard", "-t", mime, "-i", abs_path]
                subprocess.run(cmd, check=True)
                success = True

        except FileNotFoundError:
            messagebox.showwarning("Dependency Missing", "Clipboard command not found.\nOn Linux, please ensure 'xclip' is installed (sudo apt install xclip).", parent=parent_window)
            return
        except Exception as e:
            print(f"Clipboard error: {e}")
        
        if success:
            messagebox.showinfo("Clipboard", "Image copied to clipboard!", parent=parent_window)
        else:
            messagebox.showerror("Error", "Could not copy image to clipboard on this operating system.", parent=parent_window)

if __name__ == "__main__":
    app = VisualSearchApp()
    app.mainloop()