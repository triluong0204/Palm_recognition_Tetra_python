import os
from smtplib import LMTP
import main_windows
import setup
import ptkhac


if __name__ == "__main__":
    import sys
    app = main_windows.QtWidgets.QApplication(sys.argv)
    MainWindow = main_windows.QtWidgets.QMainWindow()
    ui = main_windows.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())