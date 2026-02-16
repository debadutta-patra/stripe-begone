import threading
import json
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QSplitter,
    QScrollArea,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QStyleFactory,
    QFrame,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QIcon, QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from skimage import io

try:
    from ..em_image import em_image
except ImportError:
    from em_image import em_image
from .frames import IOFrame
from .widgets import ImageTabs
from .dialogs import InstructionsDialog


class MainWindow(QMainWindow):
    processing_finished_signal = Signal()
    processing_error_signal = Signal(str)
    progress_signal = Signal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stripe Begone")
        self.setWindowIcon(QIcon("./icon.png"))
        self.resize(1200, 900)
        self.setAcceptDrops(True)

        self.process_frame = IOFrame(self)

        # Scroll Area for Controls
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.process_frame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        self.img_panels = ImageTabs(self)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(self.scroll_area)
        main_splitter.addWidget(self.img_panels)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        self.setCentralWidget(main_splitter)
        self.centralWidget().layout()  # .setContentsMargins(0,0,0,0)

        self.create_menus()
        self.create_status_bar()

        # self.em_file = None
        self.cancel_event = threading.Event()
        self.processing_finished_signal.connect(self.on_process_finished)
        self.processing_error_signal.connect(self.on_process_error)
        self.progress_signal.connect(self.progress_bar.setValue)
        self.showing_original_in_recon = False
        self.texture_view_mode = 0

    def create_menus(self):
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open Image", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.initialize)
        file_menu.addAction(open_action)

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View Menu
        view_menu = menu_bar.addMenu("&View")
        theme_menu = view_menu.addMenu("Theme")

        available_themes = QStyleFactory.keys()
        current_style = QApplication.style().objectName()

        for theme in available_themes:
            action = QAction(theme.capitalize(), self)
            action.setCheckable(True)
            if theme == current_style:
                action.setChecked(True)
            action.triggered.connect(lambda checked, t=theme: self.change_theme(t))
            theme_menu.addAction(action)

        # Help Menu
        help_menu = menu_bar.addMenu("&Help")

        instructions_action = QAction("&Instructions", self)
        instructions_action.triggered.connect(self.show_instructions)
        help_menu.addAction(instructions_action)

    def create_status_bar(self):
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("Ready")

    def change_theme(self, theme_name):
        QApplication.setStyle(theme_name)

    def show_instructions(self):
        dlg = InstructionsDialog(self)
        dlg.exec()

    def get_figure_ax(self, container):
        """

        Retrieves existing figure/ax/canvas for a container or creates them.
        This prevents destroying and recreating widgets on every update.
        """
        if not hasattr(self, "plot_cache"):
            self.plot_cache = {}

        if container not in self.plot_cache:
            if container.layout() is None:
                container.setLayout(QVBoxLayout())

            # Clear any existing widgets (cleanup)
            while container.layout().count():
                item = container.layout().takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            fig = Figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.axis("off")

            canvas = FigureCanvasQTAgg(fig)
            toolbar = NavigationToolbar2QT(canvas, container)

            container.layout().addWidget(toolbar)
            container.layout().addWidget(canvas)

            self.plot_cache[container] = (fig, ax, canvas)

        return self.plot_cache[container]

    def get_downscale_factor(self, shape, max_dim=512):
        return max(1, (max(shape) + max_dim - 1) // max_dim)

    def show_input(self):
        fig, axis, canvas = self.get_figure_ax(self.img_panels.in_img)
        axis.clear()
        axis.axis("off")
        factor = self.get_downscale_factor(self.em_file.image.shape)
        axis.imshow(self.em_file.image[::factor, ::factor], cmap="gray")
        axis.set_title("Input Image")
        canvas.draw_idle()

    def show_input_fft(self):
        fig, axis, canvas = self.get_figure_ax(self.img_panels.in_fft)
        axis.clear()
        axis.axis("off")
        factor = self.get_downscale_factor(self.em_file.fft_shifted.shape)
        fft_sub = self.em_file.fft_shifted[::factor, ::factor]
        axis.imshow(np.log(np.abs(fft_sub)), cmap="gray")
        axis.set_title("Input Image FFT")
        canvas.draw_idle()

    def show_wedges(self, mask):
        fig, axis, canvas = self.get_figure_ax(self.img_panels.in_fft)
        axis.clear()
        axis.axis("off")
        factor = self.get_downscale_factor(self.em_file.fft_shifted.shape)
        fft_sub = self.em_file.fft_shifted[::factor, ::factor]
        axis.imshow(np.log(np.abs(fft_sub)), cmap="gray")
        axis.imshow(mask[::factor, ::factor], cmap="RdBu", alpha=0.1)
        axis.set_title("Input Image FFT")
        canvas.draw_idle()

    def show_stripe_line(self, angle):
        """
        Draws a repeating grid of red dashed lines to help align with the stripe pattern.
        """
        fig, ax, canvas = self.get_figure_ax(self.img_panels.in_img)
        ax.clear()
        ax.axis("off")

        # Show image
        h, w = self.em_file.image.shape
        factor = self.get_downscale_factor((h, w))
        ax.imshow(
            self.em_file.image[::factor, ::factor], cmap="gray", extent=[0, w, h, 0]
        )

        # Geometry calculations
        cy, cx = h // 2, w // 2
        # Use full diagonal length so lines reach edges even when rotated
        length = np.hypot(h, w)
        rad = np.deg2rad(angle)

        # Direction vector (Along the line)
        u_dx = np.cos(rad)
        u_dy = np.sin(rad)

        # Normal vector (Perpendicular, for offsetting lines)
        p_dx = -np.sin(rad)
        p_dy = np.cos(rad)

        # Grid settings
        num_lines = 4  # Lines on each side of center (Total = 9 lines)
        spacing = min(h, w) // 8  # Dynamic spacing based on image size

        for i in range(-num_lines, num_lines + 1):
            offset = i * spacing

            # Calculate offset center
            ocx = cx + offset * p_dx
            ocy = cy + offset * p_dy

            # Calculate endpoints
            x1 = ocx - length * u_dx
            y1 = ocy - length * u_dy
            x2 = ocx + length * u_dx
            y2 = ocy + length * u_dy

            # Style: Center line is slightly more opaque
            alpha = 0.6 if i == 0 else 0.25

            ax.plot([x1, x2], [y1, y2], "r--", linewidth=2, alpha=alpha)

        ax.set_title(f"Defining Stripe Angle: {angle - 180: .1f}°")

        # Lock the view to the image boundaries (prevents lines from stretching the view)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        canvas.draw_idle()

    def auto_detect_angle(self):
        if self.em_file is None:
            return

        self.status_bar.showMessage("Detecting angle...")
        QApplication.processEvents()  # Force UI update

        try:
            # 1. Run Detection
            sigma = self.process_frame.radon_sigma_spinbox.value()
            threshold = self.process_frame.peak_thresh_spinbox.value()
            angles = self.em_file.detect_stripe_angle_radon(
                sigma=sigma, threshold=threshold
            )

            # 2. Update GUI
            # Ensure "Real-Space" mode is ON so the number makes sense
            if not self.process_frame.use_real_angle.isChecked():
                self.process_frame.use_real_angle.setChecked(True)

            # Check wedge size
            wedge_size = self.process_frame.wedge_spinbox.value()
            if wedge_size == 0:
                wedge_size = 5
                self.process_frame.wedge_spinbox.setValue(5)

            kmin = self.process_frame.kmin_spinbox.value()

            angles = np.atleast_1d(angles)
            for angle in angles:
                self.em_file.add_wedge(wedge_size, angle, kmin)
                self.process_frame.wedge_list.addItem(
                    f"Wedge {len(self.em_file.wedge_size)}"
                )

            self.process_frame.process_image_button.setEnabled(True)
            self.process_frame.delete_wedge_button.setEnabled(True)
            self.process_frame.update_wedge_button.setEnabled(True)
            self.process_frame.delete_all_wedge_button.setEnabled(True)

            if len(angles) > 0:
                self.process_frame.theta_spinbox.setValue(angles[0])
                self.status_bar.showMessage(f"Detected and added {len(angles)} wedges.")

        except Exception as e:
            QMessageBox.warning(self, "Detection Failed", str(e))
            self.status_bar.showMessage("Ready")

    def plot_radon_variance(self):
        if self.em_file is None:
            return
        sigma = self.process_frame.radon_sigma_spinbox.value()
        self.status_bar.showMessage(f"Computing Radon Variance (Sigma={sigma})...")
        QApplication.processEvents()
        try:
            self.cached_theta, self.cached_variance = (
                self.em_file.compute_radon_variance(sigma)
            )
            self.update_variance_plot()
            self.img_panels.setCurrentWidget(self.img_panels.tab_hpf)
            self.status_bar.showMessage("Ready")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            self.status_bar.showMessage("Error computing variance")

    def update_variance_plot(self):
        if not hasattr(self, "cached_theta") or not hasattr(self, "cached_variance"):
            return

        fig, ax, canvas = self.get_figure_ax(self.img_panels.hpf_plot)
        ax.clear()

        theta = self.cached_theta
        variances = self.cached_variance
        threshold_val = self.process_frame.peak_thresh_spinbox.value()
        std_var = np.std(variances)
        height_thresh = std_var * threshold_val

        ax.plot(theta, variances, "b-", label="Variance")
        ax.axhline(
            y=height_thresh,
            color="r",
            linestyle="--",
            label=f"Threshold ({threshold_val:.1f} std)",
        )

        # Find peaks for visualization
        from scipy.signal import find_peaks

        peaks_indices, _ = find_peaks(variances, height=height_thresh)
        ax.plot(theta[peaks_indices], variances[peaks_indices], "rx")

        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("Radon Variance")
        ax.set_title("Stripe Detection: Variance vs Angle")
        ax.legend()
        ax.grid(True)

        if hasattr(self, "variance_click_cid"):
            canvas.mpl_disconnect(self.variance_click_cid)
        self.variance_click_cid = canvas.mpl_connect(
            "button_press_event", self.on_variance_click
        )

        canvas.draw_idle()

    def on_variance_click(self, event):
        if event.inaxes and hasattr(self, "cached_variance"):
            y = event.ydata
            std_var = np.std(self.cached_variance)
            if std_var > 0:
                new_thresh = y / std_var
                self.process_frame.peak_thresh_spinbox.setValue(new_thresh)

    def preview_radon_hpf(self):
        if self.em_file is None:
            return
        sigma = self.process_frame.radon_sigma_spinbox.value()
        self.status_bar.showMessage(f"Generating HPF Preview (Sigma={sigma})...")
        QApplication.processEvents()
        try:
            self.em_file.img_hpf = self.em_file.get_radon_hpf(sigma)
            self.show_hpf()
            self.img_panels.setCurrentWidget(self.img_panels.tab_hpf)
            self.status_bar.showMessage("Ready")
            self.plot_radon_variance()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            self.status_bar.showMessage("Error generating HPF")

    def show_hpf(self):
        fig, axis, canvas = self.get_figure_ax(self.img_panels.hpf_img)
        axis.clear()
        axis.axis("off")
        if hasattr(self.em_file, "img_hpf"):
            axis.imshow(self.em_file.img_hpf, cmap="gray")
            axis.set_title(
                f"Radon High Pass (Sigma={self.process_frame.radon_sigma_spinbox.value()})"
            )
            canvas.draw_idle()

    def show_output(self):
        fig, axis, canvas = self.get_figure_ax(self.img_panels.out_img)

        if hasattr(self, "recon_click_cid"):
            canvas.mpl_disconnect(self.recon_click_cid)

        self.recon_click_cid = canvas.mpl_connect(
            "button_press_event", self.toggle_recon_view
        )
        self.showing_original_in_recon = False
        self.update_recon_view()

    def update_recon_view(self):
        fig, axis, canvas = self.get_figure_ax(self.img_panels.out_img)
        axis.clear()
        axis.axis("off")

        if self.showing_original_in_recon:
            img = self.em_file.image
            title = "Original Image (Click to Toggle)"
        else:
            img = self.em_file.img_recon
            title = "Reconstructed Image (Click to Toggle)"

        factor = self.get_downscale_factor(img.shape)
        axis.imshow(img[::factor, ::factor], cmap="gray")
        axis.set_title(title)
        canvas.draw_idle()

    def toggle_recon_view(self, event):
        if event.inaxes:
            self.showing_original_in_recon = not self.showing_original_in_recon
            self.update_recon_view()

    def show_output_fft(self):
        fig, axis, canvas = self.get_figure_ax(self.img_panels.out_fft)
        axis.clear()
        axis.axis("off")
        factor = self.get_downscale_factor(self.em_file.processed_fft.shape)
        fft_sub = self.em_file.processed_fft[::factor, ::factor]
        axis.imshow(np.log(np.abs(fft_sub)), cmap="gray")
        axis.set_title("Processed FFT")
        canvas.draw_idle()

    def show_difference(self):
        fig, axis, canvas = self.get_figure_ax(self.img_panels.diff_img)
        axis.clear()
        axis.axis("off")
        if hasattr(self.em_file, "img_diff"):
            factor = self.get_downscale_factor(self.em_file.img_diff.shape)
            limit = np.max(np.abs(self.em_file.img_diff))
            axis.imshow(
                self.em_file.img_diff[::factor, ::factor],
                cmap="gray",
                vmin=-limit,
                vmax=limit,
            )
            axis.set_title("Difference Image")
            canvas.draw_idle()

    def show_texture(self):
        fig, axis, canvas = self.get_figure_ax(self.img_panels.texture_img)

        if hasattr(self, "texture_click_cid"):
            canvas.mpl_disconnect(self.texture_click_cid)

        if hasattr(self.em_file, "img_texture"):
            self.texture_click_cid = canvas.mpl_connect(
                "button_press_event", self.toggle_texture_view
            )
            self.texture_view_mode = 0
            self.update_texture_view()
        else:
            axis.clear()
            axis.axis("off")
            canvas.draw_idle()

    def update_texture_view(self):
        fig, axis, canvas = self.get_figure_ax(self.img_panels.texture_img)
        axis.clear()
        axis.axis("off")

        if self.texture_view_mode == 0:
            img = self.em_file.img_texture
            title = "Texture Recovered Image (Click to Toggle)"
        elif self.texture_view_mode == 1:
            img = self.em_file.image
            title = "Original Image (Click to Toggle)"
        else:
            img = self.em_file.img_recon
            title = "Reconstructed Image (Click to Toggle)"

        factor = self.get_downscale_factor(img.shape)
        axis.imshow(img[::factor, ::factor], cmap="gray")
        axis.set_title(title)
        canvas.draw_idle()

    def toggle_texture_view(self, event):
        if event.inaxes:
            self.texture_view_mode = (self.texture_view_mode + 1) % 3
            self.update_texture_view()

    def show_enhanced(self):
        fig, axis, canvas = self.get_figure_ax(self.img_panels.enh_img)
        axis.clear()
        axis.axis("off")
        factor = self.get_downscale_factor(self.em_file.img_enhanced.shape)
        axis.imshow(self.em_file.img_enhanced[::factor, ::factor], cmap="gray")
        axis.set_title("Reconstructed and Contrast Adjusted Image")
        canvas.draw_idle()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:
                file_path = urls[0].toLocalFile()
                if file_path.lower().endswith(
                    (".tif", ".tiff", ".jpg", ".jpeg", ".png")
                ):
                    event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.load_image(file_path)

    def initialize(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.tif *.jpeg *.png *.jpg)"
        )
        if filename:
            self.load_image(filename)

    def load_image(self, filename):
        try:
            if hasattr(self, "em_file"):
                self.clear_image_displays()
            self.em_file = em_image(filename)
            self.em_file.file_path = filename
            self.status_bar.showMessage(f"Loaded: {filename}")
            self.show_input()
            self.show_input_fft()

            self.process_frame.add_wedge_button.setEnabled(True)
            self.process_frame.view_wedge_button.setEnabled(True)
            self.process_frame.wedge_list.clear()
            self.process_frame.delete_all_wedge_button.setEnabled(False)
            self.process_frame.delete_wedge_button.setEnabled(False)
            self.process_frame.update_wedge_button.setEnabled(False)
            self.process_frame.save_enhanced_button.setEnabled(False)
            self.process_frame.enhance_button.setEnabled(False)
            self.process_frame.process_image_button.setEnabled(False)
            self.process_frame.recover_texture_button.setEnabled(False)
            self.process_frame.save_texture_button.setEnabled(False)
            self.process_frame.view_ref_mask_button.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open file: {e}")

    def add_wedge(self):
        val_size = self.process_frame.wedge_spinbox.value()
        val_theta = self.process_frame.theta_spinbox.value()
        val_kmin = self.process_frame.kmin_spinbox.value()

        final_theta = val_theta
        # if self.process_frame.use_real_angle.isChecked():
        #     # Shift by 180 before saving
        #     final_theta = val_theta + 180

        self.em_file.add_wedge(val_size, final_theta, val_kmin)
        self.process_frame.wedge_list.addItem(f"Wedge {len(self.em_file.wedge_size)}")
        self.process_frame.process_image_button.setEnabled(True)
        self.process_frame.delete_wedge_button.setEnabled(True)
        self.process_frame.update_wedge_button.setEnabled(True)
        self.process_frame.delete_all_wedge_button.setEnabled(True)

    def update_wedge_list(self):
        self.process_frame.wedge_list.blockSignals(True)
        self.process_frame.wedge_list.clear()
        self.process_frame.wedge_list.addItems(
            [f"Wedge {x + 1}" for x in range(len(self.em_file.wedge_size))]
        )
        self.process_frame.wedge_list.blockSignals(False)
        if self.process_frame.wedge_list.count() > 0:
            self.process_frame.wedge_list.setCurrentIndex(
                self.process_frame.wedge_list.count() - 1
            )

    def get_wedge(self, index):
        if index >= 0 and index < len(self.em_file.wedge_size):
            self.process_frame.wedge_spinbox.blockSignals(True)
            self.process_frame.theta_spinbox.blockSignals(True)
            self.process_frame.kmin_spinbox.blockSignals(True)
            self.process_frame.wedge_slider.blockSignals(True)
            self.process_frame.theta_slider.blockSignals(True)
            self.process_frame.kmin_slider.blockSignals(True)

            self.process_frame.wedge_spinbox.setValue(self.em_file.wedge_size[index])
            self.process_frame.theta_spinbox.setValue(self.em_file.wedge_angle[index])
            self.process_frame.kmin_spinbox.setValue(self.em_file.k_min[index])

            self.process_frame.wedge_slider.setValue(
                int(self.em_file.wedge_size[index] * 10)
            )
            self.process_frame.theta_slider.setValue(
                int(self.em_file.wedge_angle[index] * 10)
            )
            self.process_frame.kmin_slider.setValue(int(self.em_file.k_min[index]))

            self.process_frame.wedge_spinbox.blockSignals(False)
            self.process_frame.theta_spinbox.blockSignals(False)
            self.process_frame.kmin_spinbox.blockSignals(False)
            self.process_frame.wedge_slider.blockSignals(False)
            self.process_frame.theta_slider.blockSignals(False)
            self.process_frame.kmin_slider.blockSignals(False)

            self.live_preview_wedge()

    def delete_wedge(self):
        current_text = self.process_frame.wedge_list.currentText()
        if current_text:
            reply = QMessageBox.question(
                self,
                "Warning",
                f"Do you want to delete '{current_text}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                index = self.process_frame.wedge_list.currentIndex()
                self.em_file.delete_mask(index)
                self.update_wedge_list()
                self.process_frame.process_image_button.setStyleSheet("")

                if len(self.em_file.wedge_size) == 0:
                    self.process_frame.update_wedge_button.setEnabled(False)
                    self.process_frame.delete_wedge_button.setEnabled(False)
                    self.process_frame.process_image_button.setEnabled(False)
                    self.process_frame.view_ref_mask_button.setEnabled(False)
                    self.process_frame.delete_all_wedge_button.setEnabled(False)
        else:
            QMessageBox.warning(
                self, "None Selected", "No wedge selected.\nPlease select a wedge"
            )

    def delete_all_wedges(self):
        reply = QMessageBox.question(
            self,
            "Warning",
            "Do you want to delete all Wedges?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.em_file.clear_mask()
            self.update_wedge_list()
            self.process_frame.process_image_button.setStyleSheet("")
            self.process_frame.update_wedge_button.setEnabled(False)
            self.process_frame.delete_wedge_button.setEnabled(False)
            self.process_frame.process_image_button.setEnabled(False)
            self.process_frame.delete_all_wedge_button.setEnabled(False)
            self.process_frame.view_ref_mask_button.setEnabled(False)

    def edit_wedge(self):
        index = self.process_frame.wedge_list.currentIndex()
        if index >= 0:
            self.em_file.wedge_size[index] = self.process_frame.wedge_spinbox.value()
            self.em_file.wedge_angle[index] = self.process_frame.theta_spinbox.value()
            self.em_file.k_min[index] = self.process_frame.kmin_spinbox.value()
            self.process_frame.process_image_button.setStyleSheet("")
        else:
            QMessageBox.warning(
                self, "None Selected", "No wedge selected.\nPlease select a wedge"
            )

    def view_wedge(self):
        if len(self.em_file.wedge_size) > 0:
            mask = self.em_file.get_combined_mask()
            self.show_wedges(mask)
            self.show_input()
        else:
            self.live_preview_wedge()

    def live_preview_wedge(self):
        if self.em_file is None:
            return
        val_size = self.process_frame.wedge_spinbox.value()
        val_theta = self.process_frame.theta_spinbox.value()
        val_kmin = self.process_frame.kmin_spinbox.value()

        if self.process_frame.use_real_angle.isChecked():
            # 1. Update the Input View with the Line
            # self.show_input()
            self.show_stripe_line(val_theta + 180)

            # 2. Update the FFT View with the Mask
            # Shift by 180 degrees as requested
            # converted_theta = val_theta + 180
            # mask = self.em_file.create_mask(val_size, val_theta, val_kmin)
        else:
            # Legacy Mode: Clear the red line, just show mask
            self.show_input()
        mask = self.em_file.create_mask(val_size, val_theta, val_kmin)

        self.show_wedges(mask)

    def view_reference_mask(self):
        if (
            self.em_file
            and hasattr(self.em_file, "reference_mask")
            and self.em_file.reference_mask is not None
        ):
            self.show_wedges(self.em_file.reference_mask)
        else:
            QMessageBox.information(
                self,
                "Info",
                "Reference mask not available. Please run Weighted L1 processing first.",
            )

    def cancel_processing(self):
        self.cancel_event.set()
        self.status_bar.showMessage("Cancelling...")
        self.process_frame.cancel_button.setEnabled(False)

    def process_image(self):
        self.status_bar.showMessage("Processing...")
        self.progress_bar.setRange(0, 100)
        self.process_frame.process_image_button.setEnabled(False)
        self.process_frame.method_combo.setEnabled(False)
        self.process_frame.wedge_params_group.setEnabled(False)
        self.process_frame.wedge_mgmt_group.setEnabled(False)
        self.cancel_event.clear()
        self.process_frame.cancel_button.setEnabled(True)

        combo_text = self.process_frame.method_combo.currentText()
        if "TV" in combo_text:
            method = "tv"
        elif "POCS + L1" in combo_text:
            method = "pocs_l1"
        elif "FISTA-POCS" in combo_text:
            method = "fista_pocs"
        elif "POCS" in combo_text:
            method = "pocs"
        elif "FISTA" in combo_text:
            method = "fista"
        elif "Zero" in combo_text:
            method = "zero"
        elif "Weighted L1" in combo_text:
            method = "weighted_l1"
        else:
            method = "knn"
        tv_iter = self.process_frame.tv_iter_spinbox.value()
        tv_weight = self.process_frame.tv_weight_spinbox.value()
        safety_factor = self.process_frame.safety_spinbox.value()
        remove_wedge = self.process_frame.remove_wedge_checkbox.isChecked()
        thread = threading.Thread(
            target=self.run_processing,
            args=(method, tv_iter, tv_weight, safety_factor, remove_wedge),
        )
        thread.start()

    def run_processing(self, method, tv_iter, tv_weight, safety_factor, remove_wedge):
        try:

            def progress_callback(val):
                self.progress_signal.emit(val)

            self.em_file.process_image(
                method=method,
                tv_iter=tv_iter,
                tv_weight=tv_weight,
                safety_factor=safety_factor,
                remove_wedge=remove_wedge,
                callback=progress_callback,
                cancel_event=self.cancel_event,
            )
            self.em_file.reconstruct_image()
            self.processing_finished_signal.emit()
        except InterruptedError:
            self.processing_error_signal.emit("Processing cancelled.")
        except Exception as e:
            self.processing_error_signal.emit(str(e))

    @Slot()
    def on_process_finished(self):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.status_bar.showMessage("Processing Complete")
        self.process_frame.process_image_button.setStyleSheet(
            "background-color: #5cb85c; color: white;"
        )
        self.process_frame.process_image_button.setEnabled(True)
        self.process_frame.method_combo.setEnabled(True)
        self.process_frame.wedge_params_group.setEnabled(True)
        self.process_frame.wedge_mgmt_group.setEnabled(True)
        self.process_frame.cancel_button.setEnabled(False)
        self.process_frame.save_image.setEnabled(True)
        self.process_frame.enhance_button.setEnabled(True)
        self.process_frame.recover_texture_button.setEnabled(True)

        combo_text = self.process_frame.method_combo.currentText()
        if (
            "Weighted L1" in combo_text
            or "POCS + L1" in combo_text
        ):
            self.process_frame.view_ref_mask_button.setEnabled(True)
        else:
            self.process_frame.view_ref_mask_button.setEnabled(False)

        self.view_processed()

    @Slot(str)
    def on_process_error(self, error_msg):
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Processing Failed")
        self.process_frame.process_image_button.setEnabled(True)
        self.process_frame.method_combo.setEnabled(True)
        self.process_frame.wedge_params_group.setEnabled(True)
        self.process_frame.wedge_mgmt_group.setEnabled(True)
        self.process_frame.cancel_button.setEnabled(False)

        if error_msg == "Processing cancelled.":
            self.status_bar.showMessage("Processing Cancelled")
            return

        QMessageBox.critical(
            self, "Processing Error", f"An error occurred:\n{error_msg}"
        )

    def view_processed(self):
        self.show_output()
        self.show_output_fft()
        self.show_difference()

    def clear_image_displays(self):
        """
        self.clear_figure_ax(self.img_panels.in_img)
        self.clear_figure_ax(self.img_panels.in_fft)
        self.clear_figure_ax(self.img_panels.out_img)
        self.clear_figure_ax(self.img_panels.out_fft)
        self.clear_figure_ax(self.img_panels.diff_img)
        self.clear_figure_ax(self.img_panels.texture_img)
        self.clear_figure_ax(self.img_panels.enh_img)
        self.clear_figure_ax(self.img_panels.hpf_img)
        self.clear_figure_ax(self.img_panels.hpf_plot)
        """

    def save_processed(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "TIFF file (*.tif);;JPEG file (*.jpg);;PNG file (*.png)",
        )
        if filename:
            try:
                self.generate_report(filename)
                io.imsave(filename, np.uint8(self.em_file.img_recon))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file: {e}")

    def recover_texture(self):
        self.em_file.recover_texture(self.process_frame.sigma_spinbox.value())
        self.show_texture()
        self.process_frame.save_texture_button.setEnabled(True)
        self.img_panels.setCurrentWidget(self.img_panels.tab_texture)

    def save_texture(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "TIFF file (*.tif);;JPEG file (*.jpg);;PNG file (*.png)",
        )
        if filename:
            try:
                io.imsave(filename, np.uint8(self.em_file.img_texture))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file: {e}")

    def adjust_contrast(self):
        self.em_file.enhance_contrast(
            self.process_frame.p1_spinbox.value(), self.process_frame.p2_spinbox.value()
        )
        self.show_enhanced()
        self.process_frame.save_enhanced_button.setEnabled(True)

    def save_enhanced(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "TIFF file (*.tif);;JPEG file (*.jpg);;PNG file (*.png)",
        )
        if filename:
            try:
                io.imsave(filename, np.uint8(self.em_file.img_enhanced))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file: {e}")

    def generate_report(self, output_filename):
        report = {
            "input_file": self.em_file.file_path,
            "output_file": output_filename,
            "processing_parameters": {
                "method": self.process_frame.method_combo.currentText(),
                "tv_iterations": self.process_frame.tv_iter_spinbox.value(),
                "tv_weight": self.process_frame.tv_weight_spinbox.value(),
                "safety_factor": self.process_frame.safety_spinbox.value(),
                "remove_wedge": self.process_frame.remove_wedge_checkbox.isChecked(),
                "wedge_sizes": self.em_file.wedge_size,
                "wedge_angles": self.em_file.wedge_angle,
                "k_mins": self.em_file.k_min,
                "radon_sigma": self.process_frame.radon_sigma_spinbox.value(),
                "peak_thresh": self.process_frame.peak_thresh_spinbox.value(),
                "use_real_angle": self.process_frame.use_real_angle.isChecked(),
            },
        }

        report_filename = output_filename.rsplit(".", 1)[0] + "_report.json"
        with open(report_filename, "w") as f:
            json.dump(report, f, indent=4)

        self.status_bar.showMessage(f"Report saved: {report_filename}")

    def clear_figure_ax(self, container):
        if container in self.plot_cache:
            fig, ax, canvas = self.plot_cache[container]
            ax.clear()
            canvas.draw_idle()
