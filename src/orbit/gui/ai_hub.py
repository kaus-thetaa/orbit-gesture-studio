"""Credential-free dashboard for tracking AI-provider cooldowns and input requests."""
from __future__ import annotations

import math
import time

from PySide6.QtCore import QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..ai_accounts import AIAccount, AIAccountStore, PROVIDERS


class AddAccountDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add AI account")
        self.provider = QComboBox()
        self.provider.addItems(PROVIDERS)
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. Claude work")
        form = QGridLayout()
        form.addWidget(QLabel("Provider"), 0, 0)
        form.addWidget(self.provider, 0, 1)
        form.addWidget(QLabel("Account label"), 1, 0)
        form.addWidget(self.name, 1, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)


class AccountCard(QFrame):
    def __init__(self, account: AIAccount, store: AIAccountStore, on_change, parent=None) -> None:
        super().__init__(parent)
        self.account, self.store, self.on_change = account, store, on_change
        self.setObjectName("aiAccountCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        provider = QLabel(account.provider)
        provider.setObjectName("accountProvider")
        self.title = QLabel(account.name)
        self.title.setObjectName("accountTitle")
        self.status = QLabel()
        self.status.setObjectName("accountStatus")
        self.countdown = QLabel()
        self.countdown.setObjectName("accountCountdown")

        self.hours = QSpinBox()
        self.hours.setRange(0, 168)
        self.hours.setValue(5)
        self.hours.setSuffix(" h")
        self.minutes = QSpinBox()
        self.minutes.setRange(0, 59)
        self.minutes.setSuffix(" min")
        start = QPushButton("Start cooldown")
        start.clicked.connect(self._start_cooldown)
        need_input = QPushButton("Need my input")
        need_input.clicked.connect(lambda: self._set_status("input_needed"))
        available = QPushButton("Mark available")
        available.clicked.connect(lambda: self._set_status("available"))
        open_provider = QPushButton(f"Open {account.provider}")
        open_provider.clicked.connect(self._open_provider)
        remove = QPushButton("Remove")
        remove.setObjectName("removeAccount")
        remove.clicked.connect(self._remove)

        timer_row = QHBoxLayout()
        timer_row.addWidget(self.hours)
        timer_row.addWidget(self.minutes)
        timer_row.addWidget(start)
        status_row = QHBoxLayout()
        status_row.addWidget(self.status)
        status_row.addStretch()
        status_row.addWidget(self.countdown)
        buttons_row = QHBoxLayout()
        for button in (need_input, available, open_provider, remove):
            buttons_row.addWidget(button)

        layout = QVBoxLayout(self)
        layout.addWidget(provider)
        layout.addWidget(self.title)
        layout.addLayout(status_row)
        layout.addLayout(timer_row)
        layout.addLayout(buttons_row)
        self.refresh()

    def refresh(self) -> None:
        if self.account.status == "cooldown" and self.account.cooldown_until is not None:
            remaining = max(0, math.ceil(self.account.cooldown_until - time.time()))
            hours, remainder = divmod(remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.status.setText("COOLDOWN")
            self.status.setProperty("state", "cooldown")
            self.countdown.setText(f"{hours:02}:{minutes:02}:{seconds:02} remaining")
        elif self.account.status == "input_needed":
            self.status.setText("NEEDS YOUR INPUT")
            self.status.setProperty("state", "input")
            self.countdown.setText("Waiting for you")
        else:
            self.status.setText("AVAILABLE")
            self.status.setProperty("state", "available")
            self.countdown.setText("Ready when you are")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def _start_cooldown(self) -> None:
        try:
            self.store.start_cooldown(self.account.id, self.hours.value(), self.minutes.value())
            self.on_change()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid cooldown", str(exc))

    def _set_status(self, status: str) -> None:
        try:
            self.store.set_status(self.account.id, status)
            self.on_change()
        except ValueError as exc:
            QMessageBox.warning(self, "Could not update account", str(exc))

    def _open_provider(self) -> None:
        if not QDesktopServices.openUrl(QUrl(PROVIDERS[self.account.provider])):
            QMessageBox.warning(self, "Could not open provider", "Your default browser could not be opened.")

    def _remove(self) -> None:
        answer = QMessageBox.question(self, "Remove account", f"Remove {self.account.name} from this dashboard?")
        if answer == QMessageBox.StandardButton.Yes:
            self.store.remove(self.account.id)
            self.on_change()


class AIHub(QWidget):
    notification = Signal(str)

    def __init__(self, parent=None, store=None) -> None:
        super().__init__(parent)
        self.store = store or AIAccountStore()
        self.cards: list[AccountCard] = []
        self.notice = QLabel(
            "Browser sign-in and provider status are not read by ORBIT. "
            "Timers are your own reminders; provider limits still apply."
        )
        self.notice.setObjectName("aiHubNotice")
        self.notice.setWordWrap(True)
        self.notice.hide()

        heading = QLabel("Your AI workspace")
        heading.setObjectName("aiHubHeading")
        subtitle = QLabel("Track cooldowns, see when you need to respond, and open provider sites in your browser.")
        subtitle.setObjectName("aiHubSubtitle")
        add = QPushButton("+ Add account")
        add.setObjectName("addAccount")
        add.clicked.connect(self._add_account)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.addWidget(heading)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch()
        header.addWidget(add)

        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(14)
        cards_widget = QWidget()
        cards_widget.setLayout(self.grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(cards_widget)
        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.notice)
        layout.addWidget(scroll)
        self.setStyleSheet(
            """
            QWidget { color: #e8edf5; }
            #aiHubHeading { font-size: 24px; font-weight: 700; }
            #aiHubSubtitle, #accountProvider { color: #aab6c8; }
            #aiHubNotice { background: #27344a; border: 1px solid #536a8d; border-radius: 8px; padding: 10px; }
            #aiAccountCard { background: #202a39; border: 1px solid #3a4659; border-radius: 12px; }
            #accountTitle { font-size: 17px; font-weight: 600; }
            #accountStatus { font-weight: 700; padding: 4px 8px; border-radius: 8px; }
            #accountStatus[state="available"] { color: #8ce0ad; background: #173b2a; }
            #accountStatus[state="cooldown"] { color: #ffd27b; background: #493919; }
            #accountStatus[state="input"] { color: #ffb6b6; background: #4a252b; }
            #accountCountdown { color: #c3ccda; }
            QPushButton { background: #34445b; border: 1px solid #53647e; border-radius: 6px; padding: 7px 10px; }
            QPushButton:hover { background: #435873; }
            QPushButton#addAccount { background: #2f6dce; border-color: #4e8deb; font-weight: 600; }
            QPushButton#removeAccount { color: #ffc0c0; }
            QSpinBox, QLineEdit, QComboBox { background: #18212e; border: 1px solid #46546b; border-radius: 5px; padding: 5px; }
            """
        )
        self._refresh_cards()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(1000)

    def _add_account(self) -> None:
        dialog = AddAccountDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.store.add(dialog.provider.currentText(), dialog.name.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Could not add account", str(exc))
            return
        self._refresh_cards()

    def _tick(self) -> None:
        expired = self.store.expire_cooldowns()
        if expired:
            self._refresh_cards()
            for account in expired:
                self.notice.setText(f"{account.name}'s cooldown timer ended. Check the provider and continue if available.")
                self.notice.show()
                self.notification.emit(account.name)
        for card in self.cards:
            card.refresh()

    def _refresh_cards(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.cards = []
        for index, account in enumerate(self.store.accounts):
            card = AccountCard(account, self.store, self._refresh_cards)
            self.cards.append(card)
            self.grid.addWidget(card, index // 2, index % 2)
        self.grid.setRowStretch((len(self.cards) + 1) // 2, 1)
