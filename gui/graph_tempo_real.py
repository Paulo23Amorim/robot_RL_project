from pyqtgraph import PlotWidget
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout

class GraficoTempoReal(QWidget):
    def __init__(self, titulo, eixo_y):
        super().__init__()
        layout = QVBoxLayout()
        self.plot_widget = PlotWidget()
        self.plot_widget.setTitle(titulo)
        self.plot_widget.setLabel('left', eixo_y)
        self.plot_widget.setLabel('bottom', 'Episódios')
        self.curva = self.plot_widget.plot(pen=pg.mkPen('b', width=2))
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
        self.dados = []

    def atualizar(self, novo_valor):
        self.dados.append(novo_valor)
        self.curva.setData(self.dados)
