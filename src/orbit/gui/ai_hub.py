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
from ..devtools.runner import RunnerError, open_ai_chat


class AddAccountDialog(QDialog):
    def __init__(self, parent=None, account: AIAccount | None = None) -> None:
        super().__init__(parent)
        self.account = account
        self.setWindowTitle("Edit AI account" if account else "Add AI account")
        self.provider = QComboBox()
        self.provider.addItems(PROVIDERS)
        if account:
            self.provider.setCurrentText(account.provider)
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. Claude work")
        self.name.setText(account.name if account else "")
        self.chat_url = QLineEdit()
        self.chat_url.setPlaceholderText("https://claude.ai/chat/...")
        self.chat_url.setText(account.chat_url if account else "")
        self.chrome_profile = QLineEdit()
        self.chrome_profile.setPlaceholderText("Default or Profile 1")
        self.chrome_profile.setText(account.chrome_profile if account else "Default")
        self.hint = QLabel(
            "In that Chrome profile, open chrome://version and copy the final folder name from Profile Path "
            "(for example, Default or Profile 1)."
        )
        self.hint.setWordWrap(True)
        form = QGridLayout()
        form.addWidget(QLabel("Provider"), 0, 0)
        form.addWidget(self.provider, 0, 1)
        form.addWidget(QLabel("Account label"), 1, 0)
        form.addWidget(self.name, 1, 1)
        form.addWidget(QLabel("Chat link (optional)"), 2, 0)
        form.addWidget(self.chat_url, 2, 1)
        form.addWidget(QLabel("Chrome profile folder"), 3, 0)
        form.addWidget(self.chrome_profile, 3, 1)
        form.addWidget(self.hint, 4, 0, 1, 2)
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
        self.profile = QLabel(f"Chrome profile: {account.chrome_profile}")
        self.profile.setObjectName("accountProfile")
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
        open_chat = QPushButton("Open chat")
        open_chat.setEnabled(bool(account.chat_url))
        open_chat.clicked.connect(lambda: self._open_chat(account))
        edit = QPushButton("Edit")
        edit.clicked.connect(lambda: self._edit(account))
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
        for button in (need_input, available, open_chat, open_provider, edit, remove):
            buttons_row.addWidget(button)

        layout = QVBoxLayout(self)
        layout.addWidget(provider)
        layout.addWidget(self.title)
        layout.addWidget(self.profile)
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

    def _open_chat(self, account: AIAccount) -> None:
        try:
            open_ai_chat(account.chrome_profile, account.chat_url)
        except RunnerError as exc:
            QMessageBox.warning(self, "Could not open chat", str(exc))

    def _edit(self, account: AIAccount) -> None:
        dialog = AddAccountDialog(self, account)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.store.update(
                account.id,
                dialog.provider.currentText(),
                dialog.name.text(),
                dialog.chat_url.text(),
                dialog.chrome_profile.text(),
            )
            self.on_change()
        except ValueError as exc:
            QMessageBox.warning(self, "Could not update account", str(exc))

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
            QWidget { color: #ffff00; background: #000000; }
            #aiHubHeading { color: #00ff00; font-size: 24px; font-weight: 700; }
            #aiHubSubtitle, #accountProvider { color: #ffff00; }
            #aiHubNotice { color: #ffff00; background: #000000; border: 1px solid #ffff00; border-radius: 8px; padding: 10px; }
            #aiAccountCard { color: #ffff00; background: #000000; border: 1px solid #ffff00; border-radius: 12px; }
            #accountTitle { font-size: 17px; font-weight: 600; }
            #accountStatus { font-weight: 700; padding: 4px 8px; border-radius: 8px; }
            #accountStatus[state="available"] { color: #00ff00; background: #000000; border: 1px solid #00ff00; }
            #accountStatus[state="cooldown"] { color: #ffff00; background: #000000; border: 1px solid #ffff00; }
            #accountStatus[state="input"] { color: #ff0000; background: #000000; border: 1px solid #ff0000; }
            #accountCountdown { color: #ffff00; }
            QPushButton { color: #ffff00; background: #000000; border: 1px solid #ffff00; border-radius: 6px; padding: 7px 10px; }
            QPushButton:hover { color: #000000; background: #ffff00; }
            QPushButton#addAccount { color: #000000; background: #00ff00; border-color: #00ff00; font-weight: 600; }
            QPushButton#removeAccount { color: #ff0000; border-color: #ff0000; }
            QSpinBox, QLineEdit, QComboBox { color: #ffff00; background: #000000; border: 1px solid #ffff00; border-radius: 5px; padding: 5px; }
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
            self.store.add(
                dialog.provider.currentText(),
                dialog.name.text(),
                dialog.chat_url.text(),
                dialog.chrome_profile.text(),
            )
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
