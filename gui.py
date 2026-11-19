# GENAI: Gemini and Claude for UI implementation (pretty much whole file)
# Gemini - base code for GUI layout, widgets, and event handling (from previous tasks)
# Claude - adaptation for task 6: mask / label views, 4- vs 8-connectivity comparison, classification
#          overlay with legend and object numbers, grid of aligned single objects, size histogram and
#          statistics, effect of eps and k, shape comparison plots, table of object parameters (Treeview)

import math
import traceback
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mask_preparation import compute_mask
from objects_analysis import connected_component_labeling, labels_to_color, compute_areas, classify_objects
from geometry import crop_object, object_orientation, rotate_object, shape_descriptors

VIEW_LABELS = "1. Mask and labels"
VIEW_CONNECTIVITY = "1b. Compare 4- vs 8-connectivity"
VIEW_CLASSES = "2. Classification"
VIEW_ALIGNED = "3. Aligned single objects"
VIEW_STATS = "4. Size statistics"
VIEW_STUDY = "5. Effect of ε and k"
VIEW_SHAPES = "6. Shape comparison"
VIEWS = [VIEW_LABELS, VIEW_CONNECTIVITY, VIEW_CLASSES, VIEW_ALIGNED, VIEW_STATS, VIEW_STUDY, VIEW_SHAPES]

OBJECTS_DARKER = "darker"
OBJECTS_BRIGHTER = "brighter"
BACKGROUND_NONE = "None"
BACKGROUND_BLUR = "Gaussian blur"
BACKGROUND_TOPHAT = "Top-hat"

CATEGORIES = ["single", "connected", "other"]
CATEGORY_NAMES = {"single": "Single", "connected": "Connected", "other": "Other"}
# overlay colors (RGB 0..255) - as required: single green, connected red, other yellow
OVERLAY_COLORS = {"single": (0, 200, 0), "connected": (230, 0, 0), "other": (255, 220, 0)}
# plot colors (yellow is too light for lines on white, so a darker gold is used in plots)
PLOT_COLORS = {"single": "#1a9e1a", "connected": "#d62020", "other": "#c9a000"}
TABLE_TAG_COLORS = {"single": "#e3f6e3", "connected": "#fbe3e3", "other": "#fff7d1"}
OVERLAY_ALPHA = 0.45

# (key, header, width)
TABLE_COLUMNS = [
    ("label", "#", 45), ("category", "Category", 85), ("area", "Area", 65), ("perimeter", "Perimeter", 75),
    ("width", "W", 50), ("height", "H", 50), ("aspect_ratio", "Aspect ratio", 85), ("angle", "Angle [°]", 70),
    ("rectangularity", "Rectangularity", 95), ("circularity", "Circularity", 80),
]


class ObjectAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Geometric Analysis and Classification of Objects")
        self.root.geometry("1400x950")

        # --- Application state ---
        self.image_rgb = None     # I: np.ndarray, uint8, H x W x 3
        self.file_mask = None     # optional mask loaded from a file (overrides the computed one)
        self.results = {}         # everything computed by the last 'Process Image'
        self.params = None        # parameters used for self.results
        self.missing = None       # message of the first function that is not implemented yet

        # --- 1. Top Control Panel ---
        control_frame = tk.LabelFrame(root, text="Control Panel", padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))

        tk.Button(control_frame, text="Load Image", command=self.upload_image).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Load Mask (optional)", command=self.upload_mask).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Use computed mask", command=self.clear_file_mask).pack(side=tk.LEFT, padx=5)
        self.mask_source_var = tk.StringVar(value="Mask: computed")
        tk.Label(control_frame, textvariable=self.mask_source_var, fg="#555").pack(side=tk.LEFT, padx=(5, 15))

        tk.Label(control_frame, text="View:").pack(side=tk.LEFT, padx=(15, 2))
        self.view_var = tk.StringVar(value=VIEW_LABELS)
        view_dropdown = ttk.Combobox(control_frame, textvariable=self.view_var, state="readonly",
                                     width=32, values=VIEWS)
        view_dropdown.pack(side=tk.LEFT, padx=5)
        view_dropdown.bind("<<ComboboxSelected>>", lambda event: self.draw_view())

        tk.Button(control_frame, text="Process Image", command=self.process_image,
                  bg="#e0e0e0").pack(side=tk.LEFT, padx=15)

        # --- 2. Parameters ---
        params_frame = tk.LabelFrame(root, text="Parameters", padx=10, pady=5)
        params_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        mask_frame = tk.LabelFrame(params_frame, text="Mask", padx=5, pady=3)
        mask_frame.pack(side=tk.LEFT, padx=5)
        self.channel_var = tk.StringVar(value="Blue")
        self.objects_var = tk.StringVar(value=OBJECTS_DARKER)
        self.se_size_var = tk.IntVar(value=3)
        self.add_combobox(mask_frame, "Channel:", self.channel_var, ["Gray", "Blue"])
        self.add_combobox(mask_frame, "Objects:", self.objects_var, [OBJECTS_DARKER, OBJECTS_BRIGHTER])
        self.add_spinbox(mask_frame, "Cleaning SE (0 = off):", self.se_size_var, 0, 51, 1)

        # background correction (optional, task 5) - parameters shown only for the chosen method
        background_frame = tk.LabelFrame(params_frame, text="Background correction", padx=5, pady=3)
        background_frame.pack(side=tk.LEFT, padx=5)
        self.background_var = tk.StringVar(value=BACKGROUND_NONE)
        tk.Label(background_frame, text="Method:").pack(side=tk.LEFT, padx=(5, 2))
        background_dropdown = ttk.Combobox(background_frame, textvariable=self.background_var, state="readonly",
                                           width=13, values=[BACKGROUND_NONE, BACKGROUND_BLUR, BACKGROUND_TOPHAT])
        background_dropdown.pack(side=tk.LEFT, padx=(0, 8))
        background_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_background_params())
        self.blur_params = tk.Frame(background_frame)
        self.blur_sigma_var = tk.DoubleVar(value=30.0)
        self.add_spinbox(self.blur_params, "σ:", self.blur_sigma_var, 0.5, 200, 0.5)
        self.tophat_params = tk.Frame(background_frame)
        self.tophat_size_var = tk.IntVar(value=41)
        self.add_spinbox(self.tophat_params, "SE size (odd):", self.tophat_size_var, 3, 301, 2)
        self.update_background_params()

        labeling_frame = tk.LabelFrame(params_frame, text="Labeling", padx=5, pady=3)
        labeling_frame.pack(side=tk.LEFT, padx=5)
        self.connectivity_var = tk.StringVar(value="8")
        self.add_combobox(labeling_frame, "Connectivity:", self.connectivity_var, ["4", "8"])

        class_frame = tk.LabelFrame(params_frame, text="Classification", padx=5, pady=3)
        class_frame.pack(side=tk.LEFT, padx=5)
        self.eps_var = tk.DoubleVar(value=0.2)
        self.k_var = tk.DoubleVar(value=1.5)
        self.add_spinbox(class_frame, "ε:", self.eps_var, 0.01, 0.99, 0.05)
        self.add_spinbox(class_frame, "k:", self.k_var, 1.01, 10, 0.05)

        crop_frame = tk.LabelFrame(params_frame, text="Cutting out", padx=5, pady=3)
        crop_frame.pack(side=tk.LEFT, padx=5)
        self.margin_var = tk.IntVar(value=5)
        self.add_spinbox(crop_frame, "Margin [px]:", self.margin_var, 0, 100, 1)

        # --- 3. Main area: matplotlib figure (top) and table of objects (bottom) ---
        main = tk.Frame(root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=3, uniform="row")
        main.grid_rowconfigure(1, weight=1, uniform="row")

        figure_frame = tk.LabelFrame(main, text="Results")
        figure_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        self.figure = Figure(figsize=(4, 3), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        table_frame = tk.LabelFrame(main, text="Object parameters (click a header to sort)")
        table_frame.grid(row=1, column=0, sticky="nsew")
        self.table = ttk.Treeview(table_frame, columns=[key for key, _, _ in TABLE_COLUMNS],
                                  show="headings", height=6)
        for key, header, width in TABLE_COLUMNS:
            self.table.heading(key, text=header, command=lambda c=key: self.sort_table(c))
            self.table.column(key, width=width, anchor=tk.CENTER)
        for category, color in TABLE_TAG_COLORS.items():
            self.table.tag_configure(category, background=color)
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.pack(fill=tk.BOTH, expand=True)
        self.sort_state = (None, False)   # (column, descending)

        # --- 4. Status Bar ---
        self.status_var = tk.StringVar(value="Ready. Please load an image.")
        tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W,
                 padx=5, pady=3).pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------
    # Small widget helpers
    # ------------------------------------------------------------------
    def add_combobox(self, parent, label, variable, values):
        tk.Label(parent, text=label).pack(side=tk.LEFT, padx=(5, 2))
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly",
                     width=6).pack(side=tk.LEFT, padx=(0, 8))

    def add_spinbox(self, parent, label, variable, from_, to, increment):
        tk.Label(parent, text=label).pack(side=tk.LEFT, padx=(5, 2))
        tk.Spinbox(parent, from_=from_, to=to, increment=increment, textvariable=variable,
                   width=6).pack(side=tk.LEFT, padx=(0, 8))

    def update_background_params(self):
        self.blur_params.pack_forget()
        self.tophat_params.pack_forget()
        method = self.background_var.get()
        if method == BACKGROUND_BLUR:
            self.blur_params.pack(side=tk.LEFT)
        elif method == BACKGROUND_TOPHAT:
            self.tophat_params.pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def read_params(self):
        se_size = int(self.se_size_var.get())
        if se_size != 0 and (se_size < 3 or se_size % 2 == 0):
            raise ValueError("Cleaning SE size must be 0 (no morphology) or an odd integer >= 3.")

        # one parameter of the chosen background correction: σ (blur) or SE size (top-hat)
        background = self.background_var.get()
        background_param = None
        if background == BACKGROUND_BLUR:
            background_param = float(self.blur_sigma_var.get())
            if background_param <= 0:
                raise ValueError("σ must be > 0.")
        elif background == BACKGROUND_TOPHAT:
            background_param = int(self.tophat_size_var.get())
            if background_param < 3 or background_param % 2 == 0:
                raise ValueError("Top-hat SE size must be an odd integer >= 3.")

        eps = float(self.eps_var.get())
        k = float(self.k_var.get())
        if not 0 < eps < 1:
            raise ValueError("ε must be between 0 and 1.")
        if k <= 1 + eps:
            raise ValueError(f"k must be greater than 1 + ε = {1 + eps:.2f} "
                             "(otherwise single and connected criteria overlap).")

        margin = int(self.margin_var.get())
        if margin < 0:
            raise ValueError("Margin must be >= 0.")

        return {"channel": self.channel_var.get(), "dark_objects": self.objects_var.get() == OBJECTS_DARKER,
                "background": background, "background_param": background_param, "se_size": se_size,
                "connectivity": int(self.connectivity_var.get()), "eps": eps, "k": k, "margin": margin}

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def process_image(self):
        if self.image_rgb is None:
            self.status_var.set("Load an image first.")
            return
        try:
            params = self.read_params()
        except (ValueError, tk.TclError) as e:
            self.status_var.set(f"Invalid parameter: {e}")
            return

        self.params = params
        self.results = {}
        self.missing = None
        self.status_var.set("Computing...")
        self.root.update_idletasks()

        try:
            self.run_pipeline(params)
        except NotImplementedError as e:
            self.missing = str(e)
        except Exception as e:
            # an error in the analysis code: show it, keep the GUI alive, full traceback in the console
            traceback.print_exc()
            self.missing = f"error {type(e).__name__}: {e} (traceback in the console)"

        self.fill_table()
        self.draw_view()

    def run_pipeline(self, params):
        """Runs the analysis step by step; results are stored as soon as each step is done."""
        R = self.results

        # Step 0: mask (computed, or loaded from a file)
        if self.file_mask is not None:
            mask, t = self.file_mask, None
        else:
            mask, t = compute_mask(self.image_rgb, params["channel"], params["dark_objects"],
                                   params["background"], params["background_param"], params["se_size"])
        mask = np.asarray(mask)
        if mask.shape != self.image_rgb.shape[:2]:
            raise ValueError(f"mask shape {mask.shape} differs from image shape {self.image_rgb.shape[:2]}")
        R["mask"] = (mask != 0).astype(np.uint8) * 255
        R["t"] = t

        # Step 1: labeling (dict: connectivity -> (labels, n), the other one is computed on demand)
        labels, n = connected_component_labeling(R["mask"], params["connectivity"])
        R["labels"] = {params["connectivity"]: (labels, n)}
        R["n"] = n

        # Step 2: areas and classification
        R["areas"] = np.asarray(compute_areas(labels, n), dtype=np.float64)
        R["centroids"] = self.centroids(labels, n)
        categories, mean_area = classify_objects(R["areas"], params["eps"], params["k"])
        R["categories"] = list(categories)
        R["mean_area"] = float(mean_area)

        # Steps 3-4: cut out, orientation, rotation and descriptors of every object
        # (the table lists all objects; the grid and the statistics use only the single ones)
        objects = []
        for i in range(1, n + 1):
            image_crop, mask_crop = crop_object(self.image_rgb, labels, i, params["margin"])
            theta = float(object_orientation(mask_crop))
            image_rot, mask_rot = rotate_object(image_crop, mask_crop, theta)
            descriptors = shape_descriptors(mask_crop, mask_rot)
            objects.append({"label": i, "category": R["categories"][i], "angle": theta,
                            "image": image_rot, "mask": mask_rot, **descriptors})
        R["objects"] = objects

    @staticmethod
    def centroids(labels, n):
        """Centre of every object (only for placing the object numbers on the image)."""
        ys, xs = np.nonzero(labels)
        ids = labels[ys, xs]
        counts = np.bincount(ids, minlength=n + 1)
        counts[0] = 1
        cx = np.bincount(ids, weights=xs, minlength=n + 1) / counts
        cy = np.bincount(ids, weights=ys, minlength=n + 1) / counts
        return cx, cy

    def labels_for(self, connectivity):
        cache = self.results["labels"]
        if connectivity not in cache:
            cache[connectivity] = connected_component_labeling(self.results["mask"], connectivity)
        return cache[connectivity]

    def need(self, *keys):
        for key in keys:
            if key not in self.results:
                raise NotImplementedError(self.missing or "process the image first")

    def category_counts(self, categories):
        return {c: sum(1 for x in categories[1:] if x == c) for c in CATEGORIES}

    # ------------------------------------------------------------------
    # Drawing: dispatcher
    # ------------------------------------------------------------------
    def draw_view(self):
        if self.image_rgb is None:
            return
        self.figure.clear()
        try:
            if not self.results:
                self.draw_original_only()
                if self.missing:
                    self.status_var.set(f"Not implemented yet: {self.missing}")
            else:
                {
                    VIEW_LABELS: self.view_labels,
                    VIEW_CONNECTIVITY: self.view_connectivity,
                    VIEW_CLASSES: self.view_classes,
                    VIEW_ALIGNED: self.view_aligned,
                    VIEW_STATS: self.view_stats,
                    VIEW_STUDY: self.view_study,
                    VIEW_SHAPES: self.view_shapes,
                }[self.view_var.get()]()
                if self.missing:
                    self.status_var.set(f"Not implemented yet: {self.missing} - showing what is available")
        except NotImplementedError as e:
            self.status_var.set(f"Not implemented yet: {e}")
        except Exception as e:
            traceback.print_exc()
            self.status_var.set(f"Error while drawing: {type(e).__name__}: {e} (traceback in the console)")
        finally:
            self.canvas.draw_idle()

    def draw_original_only(self):
        ax = self.figure.add_subplot(111)
        self.show(ax, self.image_rgb, "Original I")

    @staticmethod
    def show(ax, array, title, cmap=None):
        ax.imshow(array, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # ------------------------------------------------------------------
    # View 1: mask and labels
    # ------------------------------------------------------------------
    def view_labels(self):
        axes = self.figure.subplots(1, 3)
        self.show(axes[0], self.image_rgb, "Original I")
        self.need("mask")
        t = self.results["t"]
        if t is None:
            source = "loaded from file"
        else:
            source = f"{self.params['channel']}, t = {t}"
            if self.params["background"] != BACKGROUND_NONE:
                source += f", {self.params['background']} ({self.params['background_param']})"
        self.show(axes[1], self.results["mask"], f"Mask M ({source})", cmap="gray")
        self.need("labels")
        conn = self.params["connectivity"]
        labels, n = self.labels_for(conn)
        self.status_var.set(f"{conn}-connectivity: N = {n} objects")
        self.show(axes[2], labels_to_color(labels, n), f"Labels L ({conn}-connectivity), N = {n}")

    def view_connectivity(self):
        self.need("mask", "labels")
        axes = self.figure.subplots(1, 3)
        self.show(axes[0], self.results["mask"], "Mask M", cmap="gray")
        texts = []
        for ax, conn in zip(axes[1:], (4, 8)):
            labels, n = self.labels_for(conn)
            sizes = np.bincount(labels.ravel())[1:]
            tiny = int((sizes < 10).sum())
            texts.append(f"{conn}-conn: N = {n} ({tiny} objects < 10 px)")
            self.show(ax, labels_to_color(labels, n), f"{conn}-connectivity: N = {n}\n({tiny} objects smaller than 10 px)")
        self.status_var.set(" | ".join(texts))

    # ------------------------------------------------------------------
    # View 2: classification overlay
    # ------------------------------------------------------------------
    def view_classes(self):
        self.need("categories")
        labels, n = self.results["labels"][self.params["connectivity"]]
        categories = self.results["categories"]
        mean_area = self.results["mean_area"]
        eps, k = self.params["eps"], self.params["k"]

        # color of every label (row 0 = background, not used)
        label_colors = np.zeros((n + 1, 3))
        for i in range(1, n + 1):
            label_colors[i] = OVERLAY_COLORS[categories[i]]
        overlay = self.image_rgb.astype(np.float64)
        objects = labels > 0
        overlay[objects] = (1 - OVERLAY_ALPHA) * overlay[objects] + OVERLAY_ALPHA * label_colors[labels[objects]]

        ax = self.figure.add_subplot(111)
        ax.imshow(overlay.astype(np.uint8))
        ax.axis("off")

        cx, cy = self.results["centroids"]
        for i in range(1, n + 1):
            ax.text(cx[i], cy[i], str(i), fontsize=7, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

        counts = self.category_counts(categories)
        handles = [Patch(color=np.array(OVERLAY_COLORS[c]) / 255, label=f"{CATEGORY_NAMES[c]}: {counts[c]}")
                   for c in CATEGORIES]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
        ax.set_title(f"N = {n}, Ā = {mean_area:.0f} px | single: {mean_area * (1 - eps):.0f}–"
                     f"{mean_area * (1 + eps):.0f} px | connected: > {k * mean_area:.0f} px", fontsize=9)
        self.status_var.set("  ".join(f"{CATEGORY_NAMES[c]}: {counts[c]}" for c in CATEGORIES)
                            + "  (area-based classification is approximate)")

    # ------------------------------------------------------------------
    # View 3: grid of aligned single objects
    # ------------------------------------------------------------------
    def view_aligned(self):
        self.need("objects")
        singles = [o for o in self.results["objects"] if o["category"] == "single"]
        if not singles:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No single objects", ha="center", va="center")
            ax.axis("off")
            return

        # all tiles have the same size, so every object is shown at the same scale (proportions kept)
        tile_h = max(o["mask"].shape[0] for o in singles)
        tile_w = max(o["mask"].shape[1] for o in singles)
        cols = min(8, len(singles))
        rows = math.ceil(len(singles) / cols)
        axes = self.figure.subplots(rows, cols, squeeze=False)

        for ax in axes.ravel():
            ax.axis("off")
        for ax, obj in zip(axes.ravel(), singles):
            ax.imshow(self.make_tile(obj, tile_h, tile_w))
            ax.set_title(f"#{obj['label']}  θ = {obj['angle']:.1f}°", fontsize=8)
        self.status_var.set(f"{len(singles)} single objects after orientation normalization")

    @staticmethod
    def make_tile(obj, tile_h, tile_w):
        image = np.asarray(obj["image"], dtype=np.float64)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        mask = np.asarray(obj["mask"]) > 0
        h, w = mask.shape
        tile = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
        y0, x0 = (tile_h - h) // 2, (tile_w - w) // 2
        piece = np.where(mask[..., None], np.clip(image, 0, 255), 255).astype(np.uint8)
        tile[y0:y0 + h, x0:x0 + w] = piece
        return tile

    # ------------------------------------------------------------------
    # View 4: size statistics
    # ------------------------------------------------------------------
    def view_stats(self):
        self.need("categories")
        areas = self.results["areas"][1:]
        categories = np.array(self.results["categories"][1:])
        mean_area = self.results["mean_area"]
        eps, k = self.params["eps"], self.params["k"]

        ax_hist, ax_text = self.figure.subplots(1, 2, width_ratios=[3, 1])
        bins = np.linspace(0, areas.max() if areas.size else 1, 31)
        data = [areas[categories == c] for c in CATEGORIES]
        ax_hist.hist(data, bins=bins, stacked=True, color=[PLOT_COLORS[c] for c in CATEGORIES],
                     label=[CATEGORY_NAMES[c] for c in CATEGORIES])
        for value, text, style in [(mean_area, "Ā", "-"), (mean_area * (1 - eps), "Ā(1−ε)", "--"),
                                   (mean_area * (1 + eps), "Ā(1+ε)", "-."), (k * mean_area, "kĀ", ":")]:
            ax_hist.axvline(value, color="black", linestyle=style, linewidth=1, label=f"{text} = {value:.0f}")
        ax_hist.set_xlabel("Area [px]")
        ax_hist.set_ylabel("Number of objects")
        ax_hist.set_title("Distribution of object areas", fontsize=9)
        ax_hist.legend(fontsize=8)

        singles = areas[categories == "single"]
        ax_text.axis("off")
        if singles.size:
            lines = ["Single objects (area [px])", "",
                     f"count  {singles.size:>9d}", f"mean   {singles.mean():>9.1f}",
                     f"std    {singles.std():>9.1f}", f"min    {singles.min():>9.0f}",
                     f"max    {singles.max():>9.0f}"]
            self.status_var.set(f"Single objects: mean {singles.mean():.1f}, std {singles.std():.1f}, "
                                f"min {singles.min():.0f}, max {singles.max():.0f}")
        else:
            lines = ["No single objects"]
        ax_text.text(0, 1, "\n".join(lines), family="monospace", fontsize=10, va="top")

    # ------------------------------------------------------------------
    # View 5: effect of eps and k on the classification
    # ------------------------------------------------------------------
    def view_study(self):
        self.need("categories")
        eps, k = self.params["eps"], self.params["k"]
        ax_eps, ax_k = self.figure.subplots(1, 2, sharey=True)

        eps_values = np.round(np.arange(0.05, 0.96, 0.05), 2)
        k_values = np.round(np.arange(1.05, 4.01, 0.05), 2)
        self.plot_counts(ax_eps, eps_values, lambda e: (e, k), valid=lambda e: k > 1 + e)
        self.plot_counts(ax_k, k_values, lambda kk: (eps, kk), valid=lambda kk: kk > 1 + eps)

        ax_eps.axvline(eps, color="gray", linestyle="--", linewidth=1)
        ax_eps.set_xlabel("ε")
        ax_eps.set_ylabel("Number of objects")
        ax_eps.set_title(f"Effect of ε (k = {k}; only ε < k − 1)", fontsize=9)
        ax_k.axvline(k, color="gray", linestyle="--", linewidth=1)
        ax_k.set_xlabel("k")
        ax_k.set_title(f"Effect of k (ε = {eps}; only k > 1 + ε)", fontsize=9)
        ax_eps.legend(fontsize=8)
        self.status_var.set("Number of objects in each category for different ε and k (dashed line = current value)")

    def plot_counts(self, ax, values, make_params, valid):
        counts = {c: [] for c in CATEGORIES}
        for value in values:
            if not valid(value):
                for c in CATEGORIES:
                    counts[c].append(np.nan)
                continue
            categories, _ = classify_objects(self.results["areas"], *make_params(value))
            for c, count in self.category_counts(list(categories)).items():
                counts[c].append(count)
        for c in CATEGORIES:
            ax.plot(values, counts[c], marker=".", color=PLOT_COLORS[c], label=CATEGORY_NAMES[c])
        ax.grid(alpha=0.3)

    # ------------------------------------------------------------------
    # View 6: shape comparison (descriptors of all objects)
    # ------------------------------------------------------------------
    def view_shapes(self):
        self.need("objects")
        objects = self.results["objects"]
        ax_rc, ax_ac = self.figure.subplots(1, 2)
        for ax, x_key, y_key in [(ax_rc, "circularity", "rectangularity"), (ax_ac, "circularity", "aspect_ratio")]:
            for c in CATEGORIES:
                group = [o for o in objects if o["category"] == c]
                ax.scatter([o[x_key] for o in group], [o[y_key] for o in group], s=25,
                           color=PLOT_COLORS[c], label=CATEGORY_NAMES[c])
            for o in objects:
                ax.annotate(str(o["label"]), (o[x_key], o[y_key]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel(x_key.capitalize())
            ax.set_ylabel(y_key.replace("_", " ").capitalize())
            ax.grid(alpha=0.3)
        ax_rc.set_title("Rectangularity vs circularity", fontsize=9)
        ax_ac.set_title("Aspect ratio vs circularity", fontsize=9)
        ax_rc.legend(fontsize=8)
        self.status_var.set("Shape descriptors of all objects (numbers = labels)")

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    @staticmethod
    def format_value(value):
        if isinstance(value, str):
            return value
        value = float(value)
        return f"{value:.0f}" if value.is_integer() else f"{value:.3f}"

    def fill_table(self):
        self.table.delete(*self.table.get_children())
        self.sort_state = (None, False)
        if "categories" not in self.results:
            return
        objects = {o["label"]: o for o in self.results.get("objects", [])}
        for i in range(1, self.results["n"] + 1):
            category = self.results["categories"][i]
            row = objects.get(i, {"label": i, "category": category, "area": self.results["areas"][i]})
            values = [self.format_value(row[key]) if key in row else "–" for key, _, _ in TABLE_COLUMNS]
            values[1] = CATEGORY_NAMES[category]
            self.table.insert("", tk.END, values=values, tags=(category,))

    def sort_table(self, column):
        last_column, descending = self.sort_state
        descending = not descending if column == last_column else False
        self.sort_state = (column, descending)

        def key(item):
            value = self.table.set(item, column)
            try:
                return (0, float(value))
            except ValueError:
                return (1, value)

        items = sorted(self.table.get_children(""), key=key, reverse=descending)
        for index, item in enumerate(items):
            self.table.move(item, "", index)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        self.image_rgb = np.array(Image.open(file_path).convert("RGB"))
        self.file_mask = None
        self.mask_source_var.set("Mask: computed")
        self.results = {}
        self.missing = None
        self.fill_table()
        self.draw_view()
        h, w = self.image_rgb.shape[:2]
        self.status_var.set(f"Loaded: {file_path} ({w}x{h}) - set the parameters and click 'Process Image'")

    def upload_mask(self):
        if self.image_rgb is None:
            self.status_var.set("Load an image first.")
            return
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.bmp *.tif *.tiff *.jpg *.jpeg")])
        if file_path:
            self.load_mask(file_path)

    def load_mask(self, file_path):
        # 8-bit mask: 0 = background, any other value = object
        mask = np.array(Image.open(file_path).convert("L"))
        if mask.shape != self.image_rgb.shape[:2]:
            self.status_var.set(f"Mask size {mask.shape[::-1]} differs from image size "
                                f"{self.image_rgb.shape[1::-1]} - mask not loaded.")
            return
        self.file_mask = (mask != 0).astype(np.uint8) * 255
        self.mask_source_var.set("Mask: from file")
        self.status_var.set(f"Mask loaded: {file_path} - click 'Process Image'")

    def clear_file_mask(self):
        self.file_mask = None
        self.mask_source_var.set("Mask: computed")
        self.status_var.set("Mask will be computed from the image - click 'Process Image'")


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectAnalysisGUI(root)
    root.mainloop()
