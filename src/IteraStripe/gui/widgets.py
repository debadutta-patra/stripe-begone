from PySide6.QtWidgets import QTabWidget, QWidget, QHBoxLayout, QVBoxLayout


class ImageTabs(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.tab_input = QWidget()
        self.tab_recon = QWidget()
        self.tab_diff = QWidget()
        self.tab_texture = QWidget()
        self.tab_enhanced = QWidget()
        self.tab_hpf = QWidget()

        self.addTab(self.tab_input, "Input")
        self.addTab(self.tab_recon, "Reconstruction")
        self.addTab(self.tab_diff, "Difference")
        self.addTab(self.tab_texture, "Texture")
        self.addTab(self.tab_enhanced, "Enhanced")
        self.addTab(self.tab_hpf, "Radon HPF")

        input_layout = QHBoxLayout(self.tab_input)
        recon_layout = QHBoxLayout(self.tab_recon)
        diff_layout = QVBoxLayout(self.tab_diff)
        texture_layout = QVBoxLayout(self.tab_texture)
        enh_layout = QVBoxLayout(self.tab_enhanced)
        hpf_layout = QVBoxLayout(self.tab_hpf)

        self.in_img = QWidget()
        self.in_fft = QWidget()
        self.out_img = QWidget()
        self.out_fft = QWidget()
        self.diff_img = QWidget()
        self.texture_img = QWidget()
        self.enh_img = QWidget()
        self.hpf_img = QWidget()
        self.hpf_plot = QWidget()

        self.in_img.setLayout(QVBoxLayout())
        self.in_fft.setLayout(QVBoxLayout())
        self.out_img.setLayout(QVBoxLayout())
        self.out_fft.setLayout(QVBoxLayout())
        self.diff_img.setLayout(QVBoxLayout())
        self.texture_img.setLayout(QVBoxLayout())
        self.enh_img.setLayout(QVBoxLayout())
        self.hpf_img.setLayout(QVBoxLayout())
        self.hpf_plot.setLayout(QVBoxLayout())

        input_layout.addWidget(self.in_img)
        input_layout.addWidget(self.in_fft)

        recon_layout.addWidget(self.out_img)
        recon_layout.addWidget(self.out_fft)

        diff_layout.addWidget(self.diff_img)

        texture_layout.addWidget(self.texture_img)

        enh_layout.addWidget(self.enh_img)

        hpf_layout.addWidget(self.hpf_img)
        hpf_layout.addWidget(self.hpf_plot)
