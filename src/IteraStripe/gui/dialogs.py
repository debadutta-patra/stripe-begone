from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton

instructions = """
1. Use the Open  File button to open and image file.
2. Adjust the Wedge size and Theta values to define the wedge.
3. Use the add wedge button to add the current values to the processing parameters.
4. You can view and edit the already added wedges using the provided drop down button.
5. Once satisfied with the parameters click on remove stripe button to strat the processing. Once the processing is complete the button should turn green.
6. The processed image should be present in the Reconstruction tab.
7. If you encounter a significant loss in contrast adjust the p1 and p2 values perform a simple contrast enhancement.

App created by Debadutta Patra
"""


class InstructionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Instructions")
        self.resize(600, 400)
        layout = QVBoxLayout(self)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setText(instructions)
        layout.addWidget(text_edit)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
