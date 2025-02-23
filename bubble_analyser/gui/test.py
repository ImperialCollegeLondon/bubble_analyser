import sys
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PySide6.QtCore import QProcess

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Restart Example")
        self.setGeometry(100, 100, 300, 200)
        
        layout = QVBoxLayout(self)
        
        self.restart_button = QPushButton("Restart")
        self.restart_button.clicked.connect(self.restart)
        layout.addWidget(self.restart_button)
        
        self.setLayout(layout)
    
    def restart(self):
        # Start a new detached process running the same Python executable with the same arguments
        QProcess.startDetached(sys.executable, sys.argv)
        # Quit the current application
        QApplication.quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())