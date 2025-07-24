import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QComboBox, QRadioButton, QButtonGroup, QDoubleSpinBox, QSpinBox
)

class RLGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Robot Factory Controller")
        self.setFixedSize(400, 350)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Fase
        self.fase_label = QLabel("Seleciona a fase:")
        self.fase_combo = QComboBox()
        self.fase_combo.addItems(["Fase 1", "Fase 2", "Fase 3"])
        layout.addWidget(self.fase_label)
        layout.addWidget(self.fase_combo)

        # Tipo de execução
        self.tipo_label = QLabel("Modo:")
        self.radio_train = QRadioButton("Treinar")
        self.radio_test = QRadioButton("Testar")
        self.radio_train.setChecked(True)
        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.radio_train)
        self.radio_group.addButton(self.radio_test)
        layout.addWidget(self.tipo_label)
        layout.addWidget(self.radio_train)
        layout.addWidget(self.radio_test)

        # Parâmetros
        self.epsilon_label = QLabel("Epsilon:")
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setValue(1.0)
        self.epsilon_spin.setSingleStep(0.01)

        self.decay_label = QLabel("Epsilon Decay:")
        self.decay_spin = QDoubleSpinBox()
        self.decay_spin.setValue(0.995)
        self.decay_spin.setSingleStep(0.001)

        self.episodes_label = QLabel("Episódios:")
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setValue(10000)
        self.episodes_spin.setMaximum(100000)

        layout.addWidget(self.epsilon_label)
        layout.addWidget(self.epsilon_spin)
        layout.addWidget(self.decay_label)
        layout.addWidget(self.decay_spin)
        layout.addWidget(self.episodes_label)
        layout.addWidget(self.episodes_spin)

        # Botão de executar
        self.btn_run = QPushButton("Iniciar")
        self.btn_run.clicked.connect(self.executar)
        layout.addWidget(self.btn_run)

        self.setLayout(layout)

    def executar(self):
        fase = self.fase_combo.currentText()
        modo = "treinar" if self.radio_train.isChecked() else "testar"
        epsilon = self.epsilon_spin.value()
        decay = self.decay_spin.value()
        episodios = self.episodes_spin.value()

        print(f"[INFO] Executando {modo} na {fase}")
        print(f"epsilon = {epsilon}, decay = {decay}, episódios = {episodios}")

        try:
            if fase == "Fase 1":
                if modo == "treinar":
                    from gui.fase1 import train
                    train.treinar(epsilon=epsilon, epsilon_decay=decay, n_episodes=episodios)
                else:
                    from fase1 import main_test
            elif fase == "Fase 2":
                if modo == "treinar":
                    from gui.fase2 import train
                    train.treinar(epsilon=epsilon, epsilon_decay=decay, n_episodes=episodios)
                else:
                    from fase2 import main_test_fase2
            elif fase == "Fase 3":
                if modo == "treinar":
                    from gui.fase3 import train
                    train.treinar(epsilon=epsilon, epsilon_decay=decay, n_episodes=episodios)
                else:
                    from fase3 import main_test_fase3_final

        except Exception as e:
            print(f"[ERRO] Falha ao executar: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RLGui()
    gui.show()
    sys.exit(app.exec_())
