from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QGroupBox,
    QGridLayout,
    QLabel,
    QDoubleSpinBox,
    QSlider,
    QPushButton,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QSizePolicy,
)
from PySide6.QtCore import Qt


class IOFrame(QFrame):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setFrameShape(QFrame.StyledPanel)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # --- File Operations (Moved to Menu) ---
        # file_group = QGroupBox("File Operations")
        # file_layout = QVBoxLayout()
        # self.input_button = QPushButton('Open file')
        # self.input_button.clicked.connect(self.controller.initialize)
        # file_layout.addWidget(self.input_button)
        # file_group.setLayout(file_layout)
        # main_layout.addWidget(file_group)

        # --- Wedge Parameters ---

        self.wedge_params_group = QGroupBox("Wedge Parameters")
        wedge_params_layout = QGridLayout()

        self.wedge_label = QLabel("Wedge Size:")
        self.wedge_spinbox = QDoubleSpinBox()
        self.wedge_spinbox.setRange(0, 180)
        self.wedge_spinbox.setSingleStep(0.2)
        self.wedge_spinbox.setValue(0)
        self.wedge_spinbox.valueChanged.connect(self.controller.live_preview_wedge)

        self.wedge_slider = QSlider(Qt.Horizontal)
        self.wedge_slider.setRange(0, 1800)
        self.wedge_slider.setValue(0)
        self.wedge_spinbox.valueChanged.connect(
            lambda v: self.wedge_slider.setValue(int(v * 10))
        )
        self.wedge_slider.valueChanged.connect(
            lambda v: self.wedge_spinbox.setValue(v / 10)
        )

        self.theta_label = QLabel("Theta:")
        self.theta_spinbox = QDoubleSpinBox()
        self.theta_spinbox.setRange(-90, 90)
        self.theta_spinbox.setSingleStep(0.2)
        self.theta_spinbox.setValue(0)
        self.theta_spinbox.valueChanged.connect(self.controller.live_preview_wedge)

        self.auto_angle_button = QPushButton("Auto")
        self.auto_angle_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # self.auto_angle_button.setFixedWidth(50)
        self.auto_angle_button.setToolTip(
            "Auto-detect stripe angle using Radon Transform"
        )
        self.auto_angle_button.clicked.connect(self.controller.auto_detect_angle)

        self.theta_slider = QSlider(Qt.Horizontal)
        self.theta_slider.setRange(-900, 900)
        self.theta_slider.setValue(0)
        self.theta_spinbox.valueChanged.connect(
            lambda v: self.theta_slider.setValue(int(v * 10))
        )
        self.theta_slider.valueChanged.connect(
            lambda v: self.theta_spinbox.setValue(v / 10)
        )

        self.kmin_label = QLabel("K_min:")
        self.kmin_spinbox = QDoubleSpinBox()
        self.kmin_spinbox.setRange(1, 200)
        self.kmin_spinbox.setSingleStep(1)
        self.kmin_spinbox.setValue(15)
        self.kmin_spinbox.valueChanged.connect(self.controller.live_preview_wedge)

        self.kmin_slider = QSlider(Qt.Horizontal)
        self.kmin_slider.setRange(1, 200)
        self.kmin_slider.setValue(15)
        self.kmin_spinbox.valueChanged.connect(
            lambda v: self.kmin_slider.setValue(int(v))
        )
        self.kmin_slider.valueChanged.connect(lambda v: self.kmin_spinbox.setValue(v))

        self.use_real_angle = QCheckBox("Define by Real-Space Stripe Angle")
        self.use_real_angle.setChecked(True)
        self.use_real_angle.toggled.connect(self.controller.live_preview_wedge)

        wedge_params_layout.setColumnStretch(1, 1)

        wedge_params_layout.addWidget(self.wedge_label, 0, 0)
        wedge_params_layout.addWidget(self.wedge_slider, 0, 1)
        wedge_params_layout.addWidget(self.wedge_spinbox, 0, 2)
        wedge_params_layout.addWidget(self.theta_label, 1, 0)
        wedge_params_layout.addWidget(self.theta_slider, 1, 1)
        wedge_params_layout.addWidget(self.theta_spinbox, 1, 2)
        wedge_params_layout.addWidget(self.auto_angle_button, 1, 3)  # Added here
        wedge_params_layout.addWidget(self.kmin_label, 2, 0)
        wedge_params_layout.addWidget(self.kmin_slider, 2, 1)
        wedge_params_layout.addWidget(self.kmin_spinbox, 2, 2)

        wedge_params_layout.addWidget(self.use_real_angle, 3, 0, 1, 2)
        self.wedge_params_group.setLayout(wedge_params_layout)

        main_layout.addWidget(self.wedge_params_group)

        self.radon_params_group = QGroupBox("Radon Parameters")
        radon_params_layout = QGridLayout()

        self.radon_sigma_label = QLabel("Radon Sigma:")
        self.radon_sigma_spinbox = QDoubleSpinBox()
        self.radon_sigma_spinbox.setRange(1, 200)
        self.radon_sigma_spinbox.setValue(2)
        self.radon_sigma_spinbox.setToolTip(
            "Sigma for High Pass Filter in Angle Detection"
        )
        self.radon_sigma_spinbox.valueChanged.connect(
            self.controller.preview_radon_hpf
        )

        self.peak_thresh_label = QLabel("Peak Thresh:")
        self.peak_thresh_spinbox = QDoubleSpinBox()
        self.peak_thresh_spinbox.setRange(0.1, 10.0)
        self.peak_thresh_spinbox.setSingleStep(0.1)
        self.peak_thresh_spinbox.setValue(2.0)
        self.peak_thresh_spinbox.setToolTip("Threshold (std dev) for peak detection")
        self.peak_thresh_spinbox.valueChanged.connect(
            self.controller.update_variance_plot
        )

        self.preview_hpf_button = QPushButton("Preview HPF")
        self.preview_hpf_button.clicked.connect(self.controller.preview_radon_hpf)

        radon_params_layout.addWidget(self.radon_sigma_label, 0, 0)
        radon_params_layout.addWidget(self.radon_sigma_spinbox, 0, 1)
        radon_params_layout.addWidget(self.peak_thresh_label, 0, 2)
        radon_params_layout.addWidget(self.peak_thresh_spinbox, 0, 3)
        radon_params_layout.addWidget(self.preview_hpf_button, 2, 0, 1, 4)

        self.radon_params_group.setLayout(radon_params_layout)

        main_layout.addWidget(self.radon_params_group)

        # --- Wedge Management ---
        self.wedge_mgmt_group = QGroupBox("Wedge Management")
        wedge_mgmt_layout = QGridLayout()

        self.wedge_list = QComboBox()
        self.wedge_list.currentIndexChanged.connect(self.controller.get_wedge)

        self.add_wedge_button = QPushButton("Add Wedge")
        self.add_wedge_button.clicked.connect(self.controller.add_wedge)
        self.add_wedge_button.setEnabled(False)

        self.update_wedge_button = QPushButton("Update Wedge")
        self.update_wedge_button.clicked.connect(self.controller.edit_wedge)
        self.update_wedge_button.setEnabled(False)

        self.delete_wedge_button = QPushButton("Delete Wedge")
        self.delete_wedge_button.clicked.connect(self.controller.delete_wedge)
        self.delete_wedge_button.setEnabled(False)
        self.delete_wedge_button.setStyleSheet(
            "background-color: #f0ad4e; color: white;"
        )

        self.delete_all_wedge_button = QPushButton("Delete All Wedges")
        self.delete_all_wedge_button.clicked.connect(self.controller.delete_all_wedges)
        self.delete_all_wedge_button.setEnabled(False)
        self.delete_all_wedge_button.setStyleSheet(
            "background-color: #d9534f; color: white;"
        )

        self.view_wedge_button = QPushButton("View Wedge")
        self.view_wedge_button.clicked.connect(self.controller.view_wedge)
        self.view_wedge_button.setEnabled(False)

        self.view_ref_mask_button = QPushButton("View Ref. Mask")
        self.view_ref_mask_button.clicked.connect(self.controller.view_reference_mask)
        self.view_ref_mask_button.setEnabled(False)

        wedge_mgmt_layout.addWidget(self.wedge_list, 0, 0, 1, 2)
        wedge_mgmt_layout.addWidget(self.add_wedge_button, 1, 0)
        wedge_mgmt_layout.addWidget(self.update_wedge_button, 1, 1)
        wedge_mgmt_layout.addWidget(
            self.delete_wedge_button,
            2,
            0,
        )
        wedge_mgmt_layout.addWidget(
            self.view_wedge_button,
            2,
            1,
        )
        wedge_mgmt_layout.addWidget(
            self.delete_all_wedge_button,
            3,
            0,
            1,
            2,
        )
        wedge_mgmt_layout.addWidget(
            self.view_ref_mask_button,
            4,
            0,
            1,
            2,
        )
        self.wedge_mgmt_group.setLayout(wedge_mgmt_layout)
        main_layout.addWidget(self.wedge_mgmt_group)

        # --- Processing ---
        process_group = QGroupBox("Processing")
        process_layout = QVBoxLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems(
            [
                "KNN Imputation",
                "TV Reconstruction",
                "POCS Reconstruction",
                "Weighted L1",
                "POCS + L1",
                "FISTA Reconstruction",
                "FISTA-POCS Reconstruction",
                "Zero Fill",
            ]
        )
        process_layout.addWidget(self.method_combo)

        tv_layout = QGridLayout()
        self.tv_iter_label = QLabel("Iterations:")
        self.tv_iter_spinbox = QSpinBox()
        self.tv_iter_spinbox.setRange(1, 100000)
        self.tv_iter_spinbox.setValue(100)

        self.tv_weight_label = QLabel("TV Weight:")
        self.tv_weight_spinbox = QDoubleSpinBox()
        self.tv_weight_spinbox.setRange(0.01, 500)
        self.tv_weight_spinbox.setValue(0.1)
        self.tv_weight_spinbox.setSingleStep(0.1)

        self.safety_label = QLabel("Safety Factor:")
        self.safety_spinbox = QDoubleSpinBox()
        self.safety_spinbox.setRange(0.0, 20.0)
        self.safety_spinbox.setSingleStep(0.1)
        self.safety_spinbox.setValue(1.1)
        self.safety_spinbox.setEnabled(False)

        self.remove_wedge_checkbox = QCheckBox("Remove Wedge")
        self.remove_wedge_checkbox.setEnabled(False)

        tv_layout.addWidget(self.tv_iter_label, 0, 0)
        tv_layout.addWidget(self.tv_iter_spinbox, 0, 1)
        tv_layout.addWidget(self.tv_weight_label, 1, 0)
        tv_layout.addWidget(self.tv_weight_spinbox, 1, 1)
        tv_layout.addWidget(self.safety_label, 2, 0)
        tv_layout.addWidget(self.safety_spinbox, 2, 1)
        tv_layout.addWidget(self.remove_wedge_checkbox, 3, 0, 1, 2)

        self.method_combo.currentTextChanged.connect(self.update_param_labels)
        self.method_combo.setCurrentText("FISTA Reconstruction")
        process_layout.addLayout(tv_layout)

        self.process_image_button = QPushButton("Remove Stripes")
        self.process_image_button.clicked.connect(self.controller.process_image)
        self.process_image_button.setEnabled(False)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.controller.cancel_processing)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setStyleSheet("background-color: #d9534f; color: white;")

        self.save_image = QPushButton("Save Processed Image")
        self.save_image.clicked.connect(self.controller.save_processed)
        self.save_image.setEnabled(False)

        process_layout.addWidget(self.process_image_button)
        process_layout.addWidget(self.cancel_button)
        process_layout.addWidget(self.save_image)
        process_group.setLayout(process_layout)
        main_layout.addWidget(process_group)

        # --- Texture Recovery ---
        texture_group = QGroupBox("Texture Recovery")
        texture_layout = QGridLayout()

        self.sigma_label = QLabel("Sigma Clip:")
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setRange(0.1, 10.0)
        self.sigma_spinbox.setSingleStep(0.1)
        self.sigma_spinbox.setValue(3.0)

        self.recover_texture_button = QPushButton("Recover Texture")
        self.recover_texture_button.clicked.connect(self.controller.recover_texture)
        self.recover_texture_button.setEnabled(False)

        self.save_texture_button = QPushButton("Save Texture Image")
        self.save_texture_button.clicked.connect(self.controller.save_texture)
        self.save_texture_button.setEnabled(False)

        texture_layout.addWidget(self.sigma_label, 0, 0)
        texture_layout.addWidget(self.sigma_spinbox, 0, 1)
        texture_layout.addWidget(self.recover_texture_button, 1, 0, 1, 2)
        texture_layout.addWidget(self.save_texture_button, 2, 0, 1, 2)
        texture_group.setLayout(texture_layout)
        main_layout.addWidget(texture_group)

        # --- Contrast Enhancement ---
        enhance_group = QGroupBox("Contrast Enhancement")
        enhance_layout = QGridLayout()

        self.p1_label = QLabel("p_1:")
        self.p1_spinbox = QDoubleSpinBox()
        self.p1_spinbox.setRange(0, 100)
        self.p1_spinbox.setSingleStep(0.1)
        self.p1_spinbox.setValue(1)

        self.p2_label = QLabel("p_2:")
        self.p2_spinbox = QDoubleSpinBox()
        self.p2_spinbox.setRange(0, 100)
        self.p2_spinbox.setSingleStep(0.1)
        self.p2_spinbox.setValue(90)

        self.enhance_button = QPushButton("Enhance Image")
        self.enhance_button.clicked.connect(self.controller.adjust_contrast)
        self.enhance_button.setEnabled(False)

        self.save_enhanced_button = QPushButton("Save Enhanced Image")
        self.save_enhanced_button.clicked.connect(self.controller.save_enhanced)
        self.save_enhanced_button.setEnabled(False)

        enhance_layout.addWidget(self.p1_label, 0, 0)
        enhance_layout.addWidget(self.p1_spinbox, 0, 1)
        enhance_layout.addWidget(self.p2_label, 1, 0)
        enhance_layout.addWidget(self.p2_spinbox, 1, 1)
        enhance_layout.addWidget(self.enhance_button, 2, 0, 1, 2)
        enhance_layout.addWidget(self.save_enhanced_button, 3, 0, 1, 2)
        enhance_group.setLayout(enhance_layout)
        main_layout.addWidget(enhance_group)

        main_layout.setSpacing(10)
        main_layout.addStretch()

    def update_param_labels(self, text):
        self.tv_weight_spinbox.setEnabled(True)
        if (
            text == "POCS + L1"
            or text == "FISTA Reconstruction"
        ):
            self.safety_spinbox.setEnabled(True)
        else:
            self.safety_spinbox.setEnabled(False)

        if text == "POCS + L1" or text == "FISTA-POCS Reconstruction":
            self.remove_wedge_checkbox.setEnabled(True)
        else:
            self.remove_wedge_checkbox.setEnabled(False)
            self.remove_wedge_checkbox.setChecked(False)

        if "TV" in text:
            self.tv_weight_label.setText("TV Weight:")
            self.tv_weight_spinbox.setToolTip("Total Variation regularization weight.")
        elif "Weighted L1" in text or "FISTA-POCS Reconstruction" in text:
            self.tv_weight_label.setText("Unused:")
            self.tv_weight_spinbox.setEnabled(False)
            self.safety_label.setText("Safety Factor:")
            self.safety_spinbox.setToolTip("Threshold multiplier (Sigma).")
            self.safety_spinbox.setEnabled(True)
        elif "POCS + L1" in text:
            self.tv_weight_label.setText("Unused:")
            self.tv_weight_spinbox.setEnabled(False)
            self.safety_label.setText("Safety Factor:")
            self.safety_spinbox.setToolTip("Threshold multiplier (Sigma).")
        elif "FISTA" in text:
            self.tv_weight_label.setText("Unused:")
            self.tv_weight_spinbox.setEnabled(False)
            self.safety_label.setText("Safety Factor:")
            self.safety_spinbox.setToolTip("Threshold multiplier (Sigma).")
        else:
            self.tv_weight_label.setText("Weight:")
            self.safety_spinbox.setEnabled(False)
            self.tv_weight_spinbox.setToolTip("")
